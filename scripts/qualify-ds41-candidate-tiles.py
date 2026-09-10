#!/usr/bin/env python3
"""Bounded GPU candidate coordinates, context tails, capture and pointer guards."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.cuda.set_device(a.device)
    lib=C.CDLL(str(a.native_lib));tile=lib.ds41rt_v41_candidate_tile
    tile.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int32,C.c_uint64,C.c_void_p];tile.restype=C.c_int32
    stream=torch.cuda.Stream();results=[]
    with torch.cuda.stream(stream):
        for rows,width,begin in [(1,1,0),(16,8,8),(80,9,16384),(4096,8,1048568),(1,16384,0),(16,16384,1048568)]:
            storage=torch.full((rows*width+2,),-7,dtype=torch.int64,device='cuda')
            first_storage=torch.full((rows+2,),-7,dtype=torch.int64,device='cuda')
            positions=storage[1:-1].reshape(rows,width);first=first_storage[1:-1]
            def launch():
                assert tile(positions.data_ptr(),first.data_ptr(),rows,width,begin,stream.cuda_stream)==0
            def check():
                expected=torch.arange(begin,begin+width,device='cuda',dtype=torch.int64)
                expected.masked_fill_(expected>=1048576,-1)
                assert torch.equal(positions,expected.expand(rows,-1)) and (first==begin).all().item()
                assert storage[0].item()==storage[-1].item()==first_storage[0].item()==first_storage[-1].item()==-7
            launch();check()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):launch()
            positions.fill_(-23);first.fill_(-23);g.replay();check()
            args=[positions.data_ptr(),first.data_ptr(),rows,width,begin,stream.cuda_stream]
            for i in [0,1]:
                for bad in [0,args[i]+1,2**64-1]:
                    v=args.copy();v[i]=bad;assert tile(*v)!=0
            assert tile(positions.data_ptr(),positions.data_ptr(),rows,width,begin,stream.cuda_stream)!=0
            for i,bads in [(2,[0,4097]),(3,[0,16385]),(4,[1,1048576,2**64-1])]:
                for bad in bads:
                    v=args.copy();v[i]=bad;assert tile(*v)!=0
            results.append(dict(rows=rows,width=width,begin=begin,exact=True,guard_bytes=True,captured_replay=True,invalid_arguments=True))
            print(f'PASS rows={rows} width={width} begin={begin}',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,gpu=torch.cuda.get_device_name(a.device),cases=results,native_library_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
