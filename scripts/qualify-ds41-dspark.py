#!/usr/bin/env python3
"""Synthetic native dSpark numerical/RNG/graph qualification; no checkpoint load."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch

V, H, K, R = 129280, 5120, 256, 16
P, I, Z = C.c_void_p, C.c_int32, C.c_size_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/'docs/ds41-reference-lock.json').read_text())
    reference_hash = hashlib.sha256((args.reference_dir/'inference/model.py').read_bytes()).hexdigest()
    assert reference_hash == lock['files']['inference/model.py']
    kernel_hash = hashlib.sha256((args.reference_dir/'inference/kernel.py').read_bytes()).hexdigest()
    assert kernel_hash == lock['files']['inference/kernel.py']
    config_path=args.reference_dir/'inference/config.json'
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert config_hash == lock['files']['inference/config.json']
    config=json.loads(config_path.read_text())
    norm_eps,hc_eps=config['norm_eps'],config['hc_eps']
    assert norm_eps == 1e-20 and hc_eps == 1e-6
    lib = C.CDLL(str(args.native_lib))

    def bind(name, types):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, I
        def checked(*values):
            status = fn(*values)
            assert status == 0, (name, status)
        return checked, fn

    router, raw_router = bind('ds41rt_v41_router', [P,P,P,P,P,P,P,P,I,I,P])
    hc_mixes, raw_hc_mixes = bind('ds41rt_v41_hc_mixes', [P,P,P,P,P,P,P,I,P])
    hc_pre, raw_hc_pre = bind('ds41rt_v41_hc_pre', [P,P,P,I,P])
    hc_post, raw_hc_post = bind('ds41rt_v41_hc_post', [P,P,P,P,P,I,P])
    confidence, raw_confidence = bind('ds41rt_v41_dspark_confidence', [P,P,P,P,I,P])
    create, _ = bind('ds41rt_v41_markov_create', [P,C.c_uint64,C.POINTER(P)])
    destroy, _ = bind('ds41rt_v41_markov_destroy', [P])
    create_head, _ = bind('ds41rt_v41_vocabulary_head_create', [P,C.c_uint64,C.POINTER(P)])
    head_projection, _ = bind('ds41rt_v41_vocabulary_head_launch', [P,P,P,P,I,P])
    norm, _ = bind('ds41rt_cuda_ds4_rmsnorm_bf16_rne_async', [P,P,P,I,I,C.c_float,P])
    markov, raw_markov = bind('ds41rt_v41_markov_launch', [P,P,P,P,I,P])
    sample, raw_sample = bind('ds41rt_v41_draft_step_rng', [P,P,P,P,P,P,I,I,P])
    gather, _ = bind('ds41rt_cuda_embedding_lookup_bf16_async', [P,P,P,Z,Z,Z,P])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = []
    for device in map(int, args.devices.split(',')):
        torch.cuda.set_device(device)
        torch.manual_seed(4100 + device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            def bf(shape):
                return (torch.randn(shape, device='cuda')*.125).bfloat16()
            def ptr(t): return t.data_ptr()
            def close(actual, expected, atol=2e-5):
                error = (actual-expected).abs().max().item()
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=atol)
                return error
            hidden, cw = bf((80,H)), bf((H+K,))
            embedding = bf((R,K))
            all_embedding = bf((80,K))
            conf = torch.empty(80, device='cuda')
            weight, table = bf((V,K)), bf((V,K))
            logits = torch.empty((R,V), device='cuda')
            workspace = torch.empty(4*1024*1024, dtype=torch.uint8, device='cuda')
            handle = P()
            create(ptr(workspace), workspace.numel(), C.byref(handle))
            graphs = []
            try:
                errors = {}
                router_errors = []
                for experts in (128, 384):
                    topk = 3 if experts == 128 else 6
                    for rows in (1, 16, 80):
                        rh, rw = bf((rows,H)), bf((experts,H))
                        rb = torch.randn(experts, device='cuda')
                        rv = torch.randn(experts, device='cuda')
                        mask = (torch.arange(rows,device='cuda') % 2).to(torch.uint8)
                        rs = torch.empty((rows,experts), device='cuda')
                        ri = torch.empty((rows,topk), dtype=torch.int32, device='cuda')
                        rr = torch.empty((rows,topk), device='cuda')
                        def route():
                            router(ptr(rh),ptr(rw),ptr(rb),ptr(rv),ptr(mask) if experts == 384 else None,
                                   ptr(rs),ptr(ri),ptr(rr),rows,experts,stream.cuda_stream)
                        def check_route():
                            scores = torch.nn.functional.softplus(rh.float() @ rw.float().T).sqrt()
                            correction = torch.where(mask[:,None].bool(),rv,rb) if experts == 384 else rb
                            selected = (scores + correction).topk(topk,dim=-1).indices
                            assert torch.equal(ri.long(),selected)
                            weights = scores.gather(-1,selected)
                            weights = weights / (weights.sum(-1,keepdim=True)+1e-20) * 1.5
                            return {'score':close(rs,scores),'weight':close(rr,weights)}
                        route()
                        error = check_route()
                        stream.synchronize()
                        rg = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(rg,stream=stream): route()
                        rh.copy_(bf((rows,H)))
                        rb.normal_()
                        rv.normal_()
                        mask.bitwise_xor_(1)
                        rg.replay()
                        graph_error = check_route()
                        stream.synchronize()
                        rg.reset()
                        # Exact ties: the native contract chooses lowest expert IDs.
                        rh.zero_(); rw.zero_(); rb.zero_(); rv.zero_()
                        route()
                        assert torch.equal(ri.long(),torch.arange(topk,device='cuda').expand(rows,topk))
                        close(rr,torch.full_like(rr,1.5/topk))
                        # Bias changes selection only, even when it dominates the scores.
                        rb.copy_(torch.arange(experts,device='cuda',dtype=torch.float32))
                        rv.copy_(-rb)
                        route()
                        check_route()
                        # Extreme finite logits exercise softplus's overflow/underflow branches.
                        rh.fill_(1); rw.fill_(-1)
                        route()
                        assert torch.equal(rs,torch.zeros_like(rs))
                        assert torch.equal(rr,torch.zeros_like(rr))
                        rw.fill_(1)
                        route()
                        check_route()
                        assert raw_router(ptr(rh),ptr(rw),ptr(rb),ptr(rv),None,
                                          ptr(rh),ptr(ri),ptr(rr),rows,experts,stream.cuda_stream) != 0
                        assert raw_router(ptr(rh),ptr(rw),ptr(rb),None,ptr(mask),
                                          ptr(rs),ptr(ri),ptr(rr),rows,experts,stream.cuda_stream) != 0
                        assert raw_router(ptr(rh),ptr(rw),ptr(rb),ptr(rv),None,
                                          ptr(rs),ptr(ri),ptr(ri),rows,experts,stream.cuda_stream) != 0
                        router_errors.append({'experts':experts,'rows':rows,
                            'max_abs_error':error,'changed_graph_error':graph_error})
                errors['router'] = router_errors
                residual, sublayer = bf((80,4,H)), bf((80,H))
                pre = torch.randn((80,4),device='cuda')
                post = torch.randn((80,4),device='cuda')
                comb = torch.randn((80,4,4),device='cuda')
                collapsed, expanded = torch.empty_like(sublayer), torch.empty_like(residual)
                def hc_sequence(rows=80):
                    hc_pre(ptr(residual),ptr(pre),ptr(collapsed),rows,stream.cuda_stream)
                    hc_post(ptr(sublayer),ptr(residual),ptr(post),ptr(comb),ptr(expanded),rows,stream.cuda_stream)
                def hc_oracle(rows=80):
                    reference_pre = (pre[:rows,:,None]*residual[:rows].float()).sum(1).bfloat16()
                    reference_post = (post[:rows,:,None]*sublayer[:rows,None,:].float()
                        +(comb[:rows,:,:,None]*residual[:rows,:,None,:].float()).sum(1)).bfloat16()
                    assert torch.equal(collapsed[:rows],reference_pre), 'mHC pre mismatch'
                    assert torch.equal(expanded[:rows],reference_post), 'mHC post mismatch'
                mix_weight = torch.randn((24,20480),device='cuda')*.01
                mix_scale = torch.tensor([.2,.3,.4],device='cuda')
                mix_base = torch.randn(24,device='cuda')*.1
                generated_pre = torch.empty((80,4),device='cuda')
                generated_post = torch.empty((80,4),device='cuda')
                generated_comb = torch.empty((80,4,4),device='cuda')
                def coefficients(rows=80):
                    hc_mixes(ptr(residual),ptr(mix_weight),ptr(mix_scale),ptr(mix_base),
                        ptr(generated_pre),ptr(generated_post),ptr(generated_comb),rows,stream.cuda_stream)
                def coefficient_oracle(rows=80):
                    x=residual[:rows].flatten(1).float()
                    projected=(x @ mix_weight.T)*torch.rsqrt(x.square().mean(-1,keepdim=True)+norm_eps)
                    reference_pre=torch.sigmoid(projected[:,:4]*mix_scale[0]+mix_base[:4])+hc_eps
                    reference_post=2*torch.sigmoid(projected[:,4:8]*mix_scale[1]+mix_base[4:8])
                    reference_comb=(projected[:,8:]*mix_scale[2]+mix_base[8:]).reshape(rows,4,4).softmax(-1)+hc_eps
                    reference_comb=reference_comb/(reference_comb.sum(-2,keepdim=True)+hc_eps)
                    for _ in range(19):
                        reference_comb=reference_comb/(reference_comb.sum(-1,keepdim=True)+hc_eps)
                        reference_comb=reference_comb/(reference_comb.sum(-2,keepdim=True)+hc_eps)
                    return max(close(generated_pre[:rows],reference_pre),close(generated_post[:rows],reference_post),
                        close(generated_comb[:rows],reference_comb))
                for rows in [1,16,80]:
                    coefficients(rows)
                    errors[f'hc_coefficients_m{rows}']=coefficient_oracle(rows)
                stream.synchronize()
                coefficient_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(coefficient_graph,stream=stream): coefficients()
                graphs.append(coefficient_graph)
                residual.copy_(bf((80,4,H)))
                coefficient_graph.replay()
                errors['hc_coefficients_graph']=coefficient_oracle()
                base_residual=residual.clone()
                for magnitude in [1e-8,1e-10,1e-12]:
                    residual.copy_(base_residual*magnitude)
                    coefficient_graph.replay()
                    errors[f'hc_coefficients_magnitude_{magnitude}']=coefficient_oracle()
                residual.zero_()
                mix_base.copy_(torch.linspace(-100,100,24,device='cuda'))
                coefficient_graph.replay()
                errors['hc_coefficients_zero_saturated']=coefficient_oracle()
                residual.copy_(bf((80,4,H)))
                assert raw_hc_mixes(ptr(residual),ptr(mix_weight),ptr(mix_scale),ptr(mix_base),
                    ptr(generated_pre),ptr(generated_pre),ptr(generated_comb),80,stream.cuda_stream) != 0
                for rows in [1,16,80]:
                    hc_sequence(rows)
                    hc_oracle(rows)
                stream.synchronize()
                hc_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(hc_graph,stream=stream): hc_sequence()
                graphs.append(hc_graph)
                residual.copy_(bf((80,4,H)))
                pre.normal_(); comb.normal_()
                hc_graph.replay()
                hc_oracle()
                assert raw_hc_pre(ptr(residual),ptr(pre),ptr(residual),80,stream.cuda_stream) != 0
                assert raw_hc_post(ptr(sublayer),ptr(residual),ptr(post),ptr(comb),ptr(residual),80,stream.cuda_stream) != 0
                for rows in [1,16,80]:
                    confidence(ptr(hidden),ptr(all_embedding),ptr(cw),ptr(conf),rows,stream.cuda_stream)
                    expected = torch.cat([hidden[:rows],all_embedding[:rows]],-1).float() @ cw.float()
                    errors[f'confidence_m{rows}'] = close(conf[:rows],expected)
                for rows in [1,3,16]:
                    markov(handle,ptr(embedding),ptr(weight),ptr(logits),rows,stream.cuda_stream)
                    errors[f'markov_m{rows}'] = close(logits[:rows],embedding[:rows].float() @ weight.float().T)
                # The shared vocabulary weight remains BF16, shared with the backbone.
                head_weight, norm_weight = bf((V,H)), bf((H,))
                normalized = torch.empty_like(hidden)
                head_output = torch.empty((80,V),device='cuda')
                head_workspace = torch.empty(4*1024*1024,dtype=torch.uint8,device='cuda')
                head_handle = P()
                create_head(ptr(head_workspace),head_workspace.numel(),C.byref(head_handle))
                head_graph = None
                try:
                    def project_head(rows=80):
                        norm(ptr(hidden),ptr(norm_weight),ptr(normalized),rows,H,norm_eps,stream.cuda_stream)
                        head_projection(head_handle,ptr(normalized),ptr(head_weight),ptr(head_output),rows,stream.cuda_stream)
                    for rows in [1,16,80]:
                        project_head(rows)
                        x = hidden[:rows].float()
                        expected_norm = (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+norm_eps)*norm_weight.float()).bfloat16()
                        errors[f'head_norm_m{rows}'] = (normalized[:rows].float()-expected_norm.float()).abs().max().item()
                        torch.testing.assert_close(normalized[:rows],expected_norm,rtol=.008,atol=.002)
                        # FP64 isolates the native dot's error from the reference FP32
                        # GEMM's different reduction order, especially near cancellation.
                        errors[f'head_projection_m{rows}'] = close(head_output[:rows],(normalized[:rows].double() @ head_weight.double().T).float())
                        errors[f'head_combined_m{rows}'] = close(head_output[:rows],expected_norm.float() @ head_weight.float().T,atol=.002)
                    stream.synchronize()
                    head_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(head_graph,stream=stream): project_head()
                    hidden.copy_(bf((80,H)))
                    head_graph.replay()
                    close(head_output,(normalized.double() @ head_weight.double().T).float())
                finally:
                    stream.synchronize()
                    if head_graph is not None: head_graph.reset()
                    destroy(head_handle)
                del head_weight, head_output, head_workspace, normalized
                assert raw_confidence(ptr(hidden),ptr(all_embedding),ptr(cw),ptr(hidden),16,stream.cuda_stream) != 0
                assert raw_markov(handle,ptr(embedding),ptr(weight),ptr(logits),17,stream.cuda_stream) != 0
                shared = torch.randn((5,R,V),device='cuda')
                adjusted = torch.empty_like(shared)
                tokens = torch.empty((6,R),dtype=torch.int32,device='cuda')
                tokens[0].copy_(torch.arange(R,device='cuda',dtype=torch.int32))
                temperatures = torch.zeros(R,device='cuda')
                rng = torch.tensor([[12345+i,1280*i] for i in range(R)],dtype=torch.int64,device='cuda')
                def sequence():
                    for position in range(5):
                        gather(ptr(table),ptr(tokens[position]),ptr(embedding),R,V,K,stream.cuda_stream)
                        markov(handle,ptr(embedding),ptr(weight),ptr(logits),R,stream.cuda_stream)
                        all_embedding[position*R:(position+1)*R].copy_(embedding)
                        sample(ptr(shared[position]),ptr(logits),ptr(rng),ptr(temperatures),
                               ptr(adjusted[position]),ptr(tokens[position+1]),R,position,stream.cuda_stream)
                    confidence(ptr(hidden),ptr(all_embedding),ptr(cw),ptr(conf),80,stream.cuda_stream)
                def oracle():
                    current = tokens[0].long()
                    expected_tokens, expected_embeddings = [], []
                    for position in range(5):
                        emb = table[current]
                        bias = emb.float() @ weight.float().T
                        close(adjusted[position],shared[position]+bias)
                        current = (shared[position]+bias).argmax(-1)
                        expected_tokens.append(current)
                        expected_embeddings.append(emb)
                    assert torch.equal(tokens[1:].long(),torch.stack(expected_tokens))
                    expected_embedding = torch.cat(expected_embeddings)
                    assert torch.equal(all_embedding,expected_embedding)
                    close(conf,torch.cat([hidden,expected_embedding],-1).float() @ cw.float())
                sequence()
                oracle()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream): sequence()
                graphs.append(graph)
                for _ in range(3):
                    shared.normal_()
                    hidden.copy_(bf((80,H)))
                    graph.replay()
                    oracle()
                # Stochastic replay and request reordering must preserve per-seed draws.
                temperatures.fill_(1)
                sequence()
                expected_stochastic = [value.clone() for value in (tokens, adjusted, conf)]
                graph.replay()
                for actual, expected in zip((tokens, adjusted, conf), expected_stochastic):
                    assert torch.equal(actual, expected)
                shared.fill_(-100)
                shared[:,:,0] = 0.0
                shared[:,:,1] = 0.0
                logits.zero_()
                def draw(position=0):
                    sample(ptr(shared[0]),ptr(logits),ptr(rng),ptr(temperatures),
                           ptr(adjusted[0]),ptr(tokens[1]),R,position,stream.cuda_stream)
                draw()
                first = tokens[1].clone()
                draw()
                assert torch.equal(first,tokens[1])
                permutation = torch.arange(R-1,-1,-1,device='cuda')
                saved_rng = rng.clone()
                rng.copy_(saved_rng[permutation])
                draw()
                assert torch.equal(tokens[1],first[permutation])
                rng.copy_(saved_rng)
                # Known two-category distribution, with all other finite logits negligible.
                shared[0,:,0] = torch.log(torch.tensor(.7)).item()
                shared[0,:,1] = torch.log(torch.tensor(.3)).item()
                samples = []
                for attempt in range(256):
                    metadata = torch.tensor([[12345+i,1280*attempt] for i in range(R)],dtype=torch.int64,device='cuda')
                    rng.copy_(metadata)
                    draw()
                    samples.append(tokens[1].clone())
                samples = torch.stack(samples)
                assert bool(((samples == 0)|(samples == 1)).all())
                frequency = (samples == 0).float().mean().item()
                assert abs(frequency-.7) < .04, frequency
                assert raw_sample(ptr(shared),ptr(logits),ptr(rng),ptr(temperatures),ptr(adjusted),ptr(tokens),R,5,stream.cuda_stream) != 0
                stream.synchronize()
                results.append({'device':device,'name':torch.cuda.get_device_name(device),
                    'max_abs_errors':errors,'greedy_five_step_graph_mutated_replays':3,
                    'shared_head_norm_projection_graph':True,
                    'router_text_vision_graph_ties_bias_extremes_guards':True,
                    'hc_pre_post_bitwise_rows':[1,16,80], 'hc_changed_input_graph_bitwise':True,
                    'rng_replay_and_request_reordering_exact':True,
                    'stochastic_five_step_graph_exact':True,
                    'two_category_samples':samples.numel(),'category_zero_frequency':frequency})
            finally:
                stream.synchronize()
                for graph in graphs: graph.reset()
                destroy(handle)
    record = {'scope':'Native synthetic numerical and five-step graph qualification; excludes Rust owner and serving integration',
        'norm_eps':norm_eps,'hc_eps':hc_eps,'reference_config_sha256':config_hash,
        'reference_revision':lock['revision'],'reference_model_sha256':reference_hash,'reference_kernel_sha256':kernel_hash,
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results,
        'not_qualified':['Rust allocation/ownership and cancellation','Full distribution/tail quality','Alternating-wave overlap','Full model execution']}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__ == '__main__': main()
