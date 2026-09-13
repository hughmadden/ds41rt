#!/usr/bin/env python3
"""Summarize recurrent dense CUDA graph nodes from native C1 Nsight SQLite exports."""
import argparse
from collections import defaultdict
import hashlib
import json
import sqlite3
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target',required=True)
parser.add_argument('--spec',required=True)
parser.add_argument('--output',required=True)
args = parser.parse_args()
from pathlib import Path
assert not Path(args.output).exists(), 'refusing to overwrite evidence'
report = {}
for arm in ['target','spec']:
    path = Path(getattr(args,arm))
    connection = sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    nodes = defaultdict(list)
    for name,stream,gx,gy,gz,duration in connection.execute('''
        SELECT s.value,k.streamId,k.gridX,k.gridY,k.gridZ,k.end-k.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        WHERE k.graphId IS NOT NULL AND s.value LIKE '%DenseGemmKernel%'
    '''):
        nodes[name,stream,gx,gy,gz].append(duration/1000)
    groups = defaultdict(list)
    identities = defaultdict(set)
    for (name,stream,gx,gy,gz),values in nodes.items():
        if len(values)<2: continue
        geometry = None
        # Native exporter geometries, disambiguated by input layout and grid.
        # Nsight gives captured nodes virtual stream IDs; requiring repeated
        # occurrences removes one-off prefill nodes from these C1 requests.
        if 'tensor00o1280111100' in name and gz==188: geometry='query_b_N32768_K1280'
        elif 'tensor00o8192111100' in name and (gy,gz) in [(2,80),(1,40)]: geometry='output_b_N5120_K8192'
        elif 'tensor000o40968111012' in name and gz==128: geometry='output_a_groups8_N1024_K4096'
        elif 'tensor00o5120111100' in name:
            if gz==20: geometry='query_a_N1280_K5120'
            elif gz==8: geometry='kv_N512_K5120'
            elif gz in [18,36]: geometry='shared_up_N2304_K5120'
        if geometry:
            groups[geometry].extend(values)
            identities[geometry].add((name,gx,gy,gz))
    assert {'query_b_N32768_K1280','output_b_N5120_K8192'} <= groups.keys()
    report[arm] = {'sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'kernels':{key:{'calls':len(values),'median_us':statistics.median(values),
            'mean_us':statistics.mean(values),'total_ms':sum(values)/1000,
            'identities':[{'name':name,'grid':[gx,gy,gz]} for name,gx,gy,gz in sorted(identities[key])]}
            for key,values in groups.items()}}
    connection.close()
report['limitations'] = ['Instrumented CUDA kernel intervals, not untraced serving latency.',
    'Only recurrent captured dense node signatures with explicitly matched layouts/grids; quantization, reduction and other kernels excluded.',
    'Speculative samples combine backbone verification and draft calls of the same geometry; not an exact six-row-only attribution.',
    'This signature map is specific to the recorded native artifact and requires review for changed exports.']
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
for arm in ['target','spec']:
    print(arm,{key:round(value['median_us'],2) for key,value in report[arm]['kernels'].items()})
