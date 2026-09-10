#!/usr/bin/env python3
"""Inventory local V4.1 shard headers and assets without loading weight payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct

def audit(snapshot):
    raw=(snapshot/'model.safetensors.index.json').read_bytes()
    index=json.loads(raw)['weight_map']
    headers=hashlib.sha256();seen=set();shards=[]
    for name in sorted(set(index.values())):
        if Path(name).name!=name:
            raise ValueError('index shard must be a basename')
        path=snapshot/name
        with path.open('rb') as f:
            size=path.stat().st_size
            prefix=f.read(8)
            if len(prefix)!=8:raise ValueError(f'{name}: truncated prefix')
            length=struct.unpack('<Q',prefix)[0]
            if length>64*1024*1024 or length>size-8:raise ValueError(f'{name}: invalid header length')
            data=f.read(length);header=json.loads(data)
        headers.update(name.encode()+b'\0'+prefix+data)
        intervals=[]
        for tensor,entry in header.items():
            if tensor=='__metadata__':continue
            if tensor in seen or index.get(tensor)!=name:raise ValueError(f'{tensor}: inconsistent index')
            seen.add(tensor)
            start,end=entry['data_offsets']
            if not 0<=start<=end<=size-8-length:raise ValueError(f'{tensor}: invalid payload extent')
            intervals.append((start,end))
        cursor=0
        for start,end in sorted(intervals):
            if start!=cursor:raise ValueError(f'{name}: overlapping or gapped payload')
            cursor=end
        if cursor!=size-8-length:raise ValueError(f'{name}: incorrect shard length')
        shards.append({'name':name,'bytes':size,'header_sha256':hashlib.sha256(data).hexdigest()})
    if seen!=set(index):raise ValueError('index contains missing tensors')
    assets={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','inference/model.py','inference/kernel.py','inference/image_processor.py','inference/config.json','inference/convert.py','inference/engram.py']:
        path=snapshot/name
        if path.is_file():assets[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'host':socket.gethostname(),'snapshot':str(snapshot),'revision':snapshot.name,'tensors':len(seen),'shards':shards,'logical_shard_bytes':sum(s['bytes'] for s in shards),'index_sha256':hashlib.sha256(raw).hexdigest(),'headers_sha256':headers.hexdigest(),'assets_sha256':assets,'weight_payload_hashes_verified':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.snapshot.expanduser().resolve()),indent=2))
if __name__=='__main__':main()
