#!/usr/bin/env python3
"""Native interleaved grouped BF16 output projection and graph qualification."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    args=p.parse_args()
    lock=json.loads((Path(__file__).resolve().parents[1]/'docs/ds41-reference-lock.json').read_text())
    reference_hash=hashlib.sha256((args.reference_dir/'inference/convert.py').read_bytes()).hexdigest()
    assert reference_hash==lock['files']['inference/convert.py']
    torch.cuda.set_device(args.device)
    torch.manual_seed(41700+args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    create=lib.ds41rt_v41_grouped_output_create; create.argtypes=[P,C.c_uint64,C.POINTER(P)];create.restype=I
    destroy=lib.ds41rt_v41_grouped_output_destroy;destroy.argtypes=[P];destroy.restype=I
    launch=lib.ds41rt_v41_grouped_output_launch;launch.argtypes=[P,P,P,P,I,P];launch.restype=I
    dequant=lib.ds41rt_v41_grouped_output_dequant;dequant.argtypes=[P,P,P,P];dequant.restype=I
    results=[];stream=torch.cuda.Stream()
    with torch.cuda.stream(stream),torch.no_grad():
        workspace=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8);handle=P()
        assert create(workspace.data_ptr(),workspace.numel()-1,C.byref(handle))!=0 and not handle.value
        assert create(workspace.data_ptr(),workspace.numel(),C.byref(handle))==0
        try:
            native_weight=(torch.randn((8192,4096),device='cuda')*.5).to(torch.float8_e4m3fn)
            scales=torch.randint(119,128,(256,128),device='cuda',dtype=torch.uint8)
            weight=torch.empty((8,1024,4096),device='cuda',dtype=torch.bfloat16)
            assert dequant(native_weight.data_ptr(),scales.data_ptr(),weight.data_ptr(),stream.cuda_stream)==0
            expected_weight=(native_weight.float().reshape(256,32,128,32)*torch.exp2(scales.float()-127)[:,None,:,None]).reshape(8,1024,4096).bfloat16()
            torch.testing.assert_close(weight,expected_weight,rtol=0,atol=0)
            assert dequant(native_weight.data_ptr(),scales.data_ptr(),native_weight.data_ptr(),stream.cuda_stream)!=0
            del expected_weight

            for rows in (1,16,80,255,1023,4095):
                x=(torch.randn((rows,8,4096),device='cuda')*.2).bfloat16();y=torch.empty((rows,8,1024),device='cuda',dtype=torch.bfloat16)
                def run(): assert launch(handle,x.data_ptr(),weight.data_ptr(),y.data_ptr(),rows,stream.cuda_stream)==0
                def check():
                    # Separate FP32 products for every group expose incorrect interleaving.
                    expected=torch.stack([x[:,g].float()@weight[g].float().T for g in range(8)],dim=1).bfloat16()
                    torch.testing.assert_close(y,expected,rtol=.008,atol=.002)
                    return {'max_abs_error':(y.float()-expected.float()).abs().max().item(),'different_bf16_elements':(y!=expected).sum().item()}
                run();initial=check()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):run()
                x.copy_((torch.randn_like(x.float())*.4).bfloat16());graph.replay();changed=check()
                x.zero_();graph.replay();assert torch.count_nonzero(y).item()==0
                assert launch(handle,x.data_ptr(),weight.data_ptr(),x.data_ptr(),rows,stream.cuda_stream)!=0
                assert launch(handle,x.data_ptr(),weight.data_ptr(),y.data_ptr(),0,stream.cuda_stream)!=0
                assert launch(handle,x.data_ptr(),weight.data_ptr(),y.data_ptr(),4097,stream.cuda_stream)!=0
                assert launch(handle,workspace.data_ptr(),weight.data_ptr(),y.data_ptr(),1,stream.cuda_stream)!=0
                results.append({'rows':rows,'initial':initial,'changed_graph':changed,'zero_exact':True,'guards':True})
                stream.synchronize();del graph
                print(f'PASS rows={rows}',flush=True)
        finally:
            stream.synchronize();assert destroy(handle)==0
    args.output.write_text(json.dumps({'reference_convert_sha256':reference_hash,'weight_dequant_exact':True,'device':args.device,'name':torch.cuda.get_device_name(args.device),'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
if __name__=='__main__':main()
