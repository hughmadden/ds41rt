#!/usr/bin/env python3
"""Native FP8 shapes and composed shared FFN against pinned reference formulas."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch

P, I = C.c_void_p, C.c_int32

class Info(C.Structure):
    _fields_ = [(name,C.c_uint32) for name in ('abi','capacity','k','n')] + [
        (name,C.c_uint64) for name in ('scratch','values','row_scales','mma_scales','weight_scales')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1]/'docs/ds41-reference-lock.json').read_text())
    for name in ('inference/model.py','inference/kernel.py'):
        assert hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest() == lock['files'][name]
    torch.cuda.set_device(args.device)
    torch.manual_seed(4150+args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(args.native_lib))
    def bind(name,types):
        fn = getattr(lib,name)
        fn.argtypes,fn.restype = types,I
        def checked(*values):
            status = fn(*values)
            assert status == 0,(name,status)
        return checked,fn
    get_info,raw_info = bind('ds41rt_v41_fp8_matrix_info',[I,I,I,C.POINTER(Info)])
    initialize,_ = bind('ds41rt_v41_fp8_matrix_initialize',[I,I,I,C.POINTER(P)])
    init_scratch,_ = bind('ds41rt_v41_fp8_initialize_scratch',[P,P,C.c_uint64,P,P])
    pack,raw_pack = bind('ds41rt_v41_fp8_matrix_pack_scales',[P,P,I,I,P])
    launch,raw_launch = bind('ds41rt_v41_fp8_launch',[P,P,P,P,P,C.c_uint64,P,P,I,P])
    swiglu,raw_swiglu = bind('ds41rt_v41_shared_swiglu',[P,P,P,I,P])
    stream = torch.cuda.Stream()
    evidence = []
    def ptr(t): return t.data_ptr()
    def random_bf(rows,k): return (torch.randn((rows,k),device='cuda')*.5).bfloat16()
    def quant_reference(x):
        rows,k = x.shape
        block = x.float().reshape(rows,k//32,32)
        amax = block.abs().amax(-1).clamp_min(1e-4)
        scales = torch.exp2(torch.ceil(torch.log2(amax * (1/448))))
        values = (block/scales[:,:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        return values.reshape(rows,k),scales
    def reference(x,weight,scales,ordered=False):
        q,s = quant_reference(x)
        if ordered:
            # Official kernel clears its MMA accumulator every K32, then adds
            # the scale-corrected block into a separate FP32 accumulator.
            acc = torch.zeros((x.shape[0],weight.shape[0]),device='cuda')
            w = weight.float()
            sf = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32,0)
            for block in range(x.shape[1]//32):
                part = q[:,block*32:(block+1)*32].float() @ w[:,block*32:(block+1)*32].T
                acc += (part*s[:,block,None])*sf[None,:,block]
            return acc.bfloat16()
        dx = (q.float().reshape(x.shape[0],-1,32)*s[:,:,None]).reshape_as(x)
        dw = weight.float()*scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32,0).repeat_interleave(32,1)
        return (dx@dw.T).bfloat16()
    def compare(actual,expected):
        error = (actual.float()-expected.float()).abs()
        torch.testing.assert_close(actual,expected,rtol=.008,atol=.002)
        return {'max_abs_error':error.max().item(),'different_bf16_elements':int((actual!=expected).sum())}
    class Projection:
        def __init__(self,capacity,k,n):
            self.info,self.handle = Info(),P()
            get_info(capacity,k,n,C.byref(self.info))
            assert (self.info.abi,self.info.capacity,self.info.k,self.info.n) == (1,capacity,k,n)
            assert self.info.weight_scales == n*k//32
            initialize(capacity,k,n,C.byref(self.handle))
            self.weight = (torch.randn((n,k),device='cuda')*.5).to(torch.float8_e4m3fn)
            self.scales = torch.randint(120,125,(n//32,k//32),device='cuda',dtype=torch.uint8)
            self.packed = torch.empty(self.info.weight_scales,device='cuda',dtype=torch.uint8)
            pack(ptr(self.scales),ptr(self.packed),k,n,stream.cuda_stream)
            expected = self.scales.reshape(n//128,4,k//128,4).permute(0,2,1,3).unsqueeze(2).expand(-1,-1,32,-1,-1).contiguous().flatten()
            assert torch.equal(self.packed,expected)
            self.scratch = torch.empty(self.info.scratch,device='cuda',dtype=torch.uint8)
            self.alpha = torch.empty(1,device='cuda')
            init_scratch(self.handle,ptr(self.scratch),self.scratch.numel(),ptr(self.alpha),stream.cuda_stream)
            self.output = torch.empty((capacity,n),device='cuda',dtype=torch.bfloat16)
        def run(self,x,output=None):
            launch(self.handle,ptr(x),ptr(self.weight),ptr(self.packed),ptr(self.scratch),
                   self.scratch.numel(),ptr(self.alpha),ptr(self.output if output is None else output),
                   x.shape[0],stream.cuda_stream)
        def check(self,x):
            q,s = quant_reference(x)
            actual_q = self.scratch[self.info.values:self.info.values+x.numel()].reshape_as(x)
            actual_s = self.scratch[self.info.row_scales:self.info.row_scales+s.numel()].view(torch.float8_e8m0fnu).float().reshape_as(s)
            assert torch.equal(actual_q,q.view(torch.uint8))
            assert torch.equal(actual_s,s)
            return compare(self.output[:x.shape[0]],reference(x,self.weight,self.scales))
    with torch.cuda.stream(stream):
        for k,n in ((6144,25600),(5120,2304),(2304,5120)):
            for capacity in (1,16,80,256,1024,4096):
                p = Projection(capacity,k,n)
                # Tail rows use the capacity's exported quantization grid and GEMM.
                rows = capacity if capacity <= 80 else capacity-1
                x = random_bf(rows,k)
                p.run(x)
                error = p.check(x)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): p.run(x)
                x.copy_(random_bf(rows,k))
                graph.replay()
                changed = p.check(x)
                stream.synchronize()
                graph.reset()
                x.zero_()
                p.run(x)
                p.check(x)
                assert torch.equal(p.output[:rows],torch.zeros_like(p.output[:rows]))
                x.copy_(random_bf(rows,k)*1e-6)
                p.run(x)
                p.check(x)
                assert raw_pack(ptr(p.scales),ptr(p.scales),k,n,stream.cuda_stream) != 0
                assert raw_launch(p.handle,ptr(x),ptr(p.weight),ptr(p.packed),ptr(p.scratch),
                                  p.scratch.numel(),ptr(p.alpha),ptr(x),rows,stream.cuda_stream) != 0
                evidence.append({'k':k,'n':n,'capacity':capacity,'rows':rows,
                                 'projection':error,'changed_graph':changed,'quantization_and_scale_packing_exact':True,'zero_input_exact':True,'tiny_input_quantization_exact':True})
                del p,x
        for rows in (1,16,80):
            gate,up,down = Projection(rows,5120,2304),Projection(rows,5120,2304),Projection(rows,2304,5120)
            x = random_bf(rows,5120)
            intermediate = torch.empty_like(gate.output)
            def execute():
                gate.run(x); up.run(x)
                swiglu(ptr(gate.output),ptr(up.output),ptr(intermediate),rows,stream.cuda_stream)
                down.run(intermediate)
            def check():
                expected_g = reference(x,gate.weight,gate.scales,ordered=True)
                expected_u = reference(x,up.weight,up.scales,ordered=True)
                gate_error,up_error = compare(gate.output,expected_g),compare(up.output,expected_u)
                g,u = expected_g.float().clamp(max=10),expected_u.float().clamp(-10,10)
                expected_i = (torch.nn.functional.silu(g)*u).bfloat16()
                local_i = (torch.nn.functional.silu(gate.output.float().clamp(max=10))*
                           up.output.float().clamp(-10,10)).bfloat16()
                assert torch.equal(intermediate,local_i)
                # Composite differences include both gate/up BF16 rounding;
                # validate every primitive above, and bound the nonlinear propagation.
                ga,ua = gate.output.float().clamp(max=10),up.output.float().clamp(-10,10)
                raw_actual = torch.nn.functional.silu(ga)*ua
                raw_expected = torch.nn.functional.silu(g)*u
                lipschitz = 1+torch.maximum(ga.abs(),g.abs())/4
                intermediate_bound = lipschitz*(ga-g).abs()*ua.abs()+torch.nn.functional.silu(g).abs()*(ua-u).abs()
                intermediate_bound += .004*(raw_actual.abs()+raw_expected.abs())+1e-6
                intermediate_delta = (intermediate.float()-expected_i.float()).abs()
                assert bool((intermediate_delta <= intermediate_bound).all())
                intermediate_error = {'max_abs_error':intermediate_delta.max().item(),
                    'different_bf16_elements':int((intermediate!=expected_i).sum()),
                    'nonlinear_propagation_bound_passed':True}
                local_down = reference(intermediate,down.weight,down.scales,ordered=True)
                down_error = compare(down.output,local_down)
                expected_down = reference(expected_i,down.weight,down.scales,ordered=True)
                # FP8 has discontinuities: validate propagation from the actual
                # BF16 intermediate, rather than silently widening an end-to-end
                # allclose threshold. Triangle inequality bounds the dense error.
                aq,asc = quant_reference(intermediate)
                eq,esc = quant_reference(expected_i)
                delta = ((aq.float().reshape(rows,-1,32)*asc[:,:,None])-
                         (eq.float().reshape(rows,-1,32)*esc[:,:,None])).abs().reshape(rows,-1)
                dw = down.weight.float()*down.scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32,0).repeat_interleave(32,1)
                propagated = delta @ dw.abs().T
                # Two BF16 rounding errors plus the separately checked native
                # down-projection tolerance; add a small FP32 summation allowance.
                bound = propagated + .004*(local_down.float().abs()+expected_down.float().abs()) + .008*local_down.float().abs()+.002
                error = (down.output.float()-expected_down.float()).abs()
                assert bool((error <= bound).all())
                rms = error.square().mean().sqrt().item()
                reference_rms = expected_down.float().square().mean().sqrt().item()
                return {'gate':gate_error,'up':up_error,'intermediate':intermediate_error,'down_same_input':down_error,
                        'end_to_end_max_abs_error':error.max().item(),'end_to_end_rms_error':rms,
                        'reference_rms':reference_rms,'quantization_propagation_bound_passed':True,
                        'different_intermediate_bf16_elements':int((intermediate!=expected_i).sum()),
                        'different_down_input_fp8_elements':int((aq.view(torch.uint8)!=eq.view(torch.uint8)).sum())}
            execute()
            error = check()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): execute()
            x.copy_(random_bf(rows,5120))
            graph.replay()
            changed = check()
            stream.synchronize()
            graph.reset()
            # Standalone activation includes both clamp boundaries and extremes.
            extremes = torch.tensor([-100,-11,-10,-1,0,1,10,11,100],device='cuda',dtype=torch.bfloat16)
            gate.output.copy_(extremes.repeat((rows*2304+8)//9)[:rows*2304].reshape(rows,2304))
            up.output.copy_(gate.output.flip(-1))
            swiglu(ptr(gate.output),ptr(up.output),ptr(intermediate),rows,stream.cuda_stream)
            expected = (torch.nn.functional.silu(gate.output.float().clamp(max=10))*up.output.float().clamp(-10,10)).bfloat16()
            assert torch.equal(intermediate,expected)
            assert raw_swiglu(ptr(gate.output),ptr(up.output),ptr(gate.output),rows,stream.cuda_stream) != 0
            evidence.append({'shared_ffn_rows':rows,'composed':error,'changed_graph':changed,'activation_extremes_bitwise':True})
        invalid = Info()
        assert raw_info(16,4096,2304,C.byref(invalid)) != 0
        assert raw_info(17,5120,2304,C.byref(invalid)) != 0
        stream.synchronize()
    record = {'scope':'Synthetic native FP8 projections and shared FFN; excludes Rust-owned composition and serving',
              'reference_revision':lock['revision'],'reference_hashes':{n:lock['files'][n] for n in ('inference/model.py','inference/kernel.py')},
              'device':args.device,'device_name':torch.cuda.get_device_name(),'device_uuid':str(torch.cuda.get_device_properties(args.device).uuid),
              'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
              'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':evidence}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__ == '__main__': main()
