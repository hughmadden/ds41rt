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
    lib = C.CDLL(str(args.native_lib))

    def bind(name, types):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, I
        def checked(*values):
            status = fn(*values)
            assert status == 0, (name, status)
        return checked, fn

    confidence, raw_confidence = bind('ds41rt_v41_dspark_confidence', [P,P,P,P,I,P])
    create, _ = bind('ds41rt_v41_markov_create', [P,C.c_uint64,C.POINTER(P)])
    destroy, _ = bind('ds41rt_v41_markov_destroy', [P])
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
                for rows in [1,16,80]:
                    confidence(ptr(hidden),ptr(all_embedding),ptr(cw),ptr(conf),rows,stream.cuda_stream)
                    expected = torch.cat([hidden[:rows],all_embedding[:rows]],-1).float() @ cw.float()
                    errors[f'confidence_m{rows}'] = close(conf[:rows],expected)
                for rows in [1,3,16]:
                    markov(handle,ptr(embedding),ptr(weight),ptr(logits),rows,stream.cuda_stream)
                    errors[f'markov_m{rows}'] = close(logits[:rows],embedding[:rows].float() @ weight.float().T)
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
                    'rng_replay_and_request_reordering_exact':True,
                    'stochastic_five_step_graph_exact':True,
                    'two_category_samples':samples.numel(),'category_zero_frequency':frequency})
            finally:
                stream.synchronize()
                for graph in graphs: graph.reset()
                destroy(handle)
    record = {'scope':'Native synthetic numerical and five-step graph qualification; excludes Rust owner and serving integration',
        'reference_revision':lock['revision'],'reference_model_sha256':reference_hash,
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results,
        'not_qualified':['Rust allocation/ownership and cancellation','Full distribution/tail quality','Alternating-wave overlap','Full model execution']}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__ == '__main__': main()
