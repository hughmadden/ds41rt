#!/usr/bin/env python3
"""Compare warm C1 ordinary text performance with sequential alternating AB/BA arms."""
import argparse
import json
from pathlib import Path
import runpy
import statistics
API=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference-url',required=True)
p.add_argument('--candidate-url',required=True)
p.add_argument('--pairs',type=int,default=4)
p.add_argument('--tokens',type=int,default=256)
p.add_argument('--allow-output-difference',action='store_true',
               help='Precision migration comparison: retain text differences, but still require equal output-token counts')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists() and a.pairs>0 and a.tokens>1
urls=dict(reference=a.reference_url,candidate=a.candidate_url)
report=dict(scope=__doc__,urls=urls,allow_output_difference=a.allow_output_difference,workloads=[],summaries=[],passed=False)
def save():a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for name,prompt in [
    ('counting','Count from 1 to 200, separated by commas. Output only the sequence.'),
    ('code','Write a complete Python 3 implementation of a thread-safe bounded LRU cache using OrderedDict and RLock. Include get, put, delete, clear, length, and unittest cases. Output only code.')]:
    body=dict(API['payload'](prompt,True),max_tokens=a.tokens)
    workload=dict(name=name,request=body,warmups={},runs=[]);report['workloads'].append(workload);save()
    for arm,url in urls.items():workload['warmups'][arm]=API['stream_case'](url,body);save()
    for index in range(a.pairs):
        order=['reference','candidate'] if index%2==0 else ['candidate','reference']
        run=dict(order=order);workload['runs'].append(run);save()
        for arm in order:run[arm]=API['stream_case'](urls[arm],body);save()
        ref,candidate=run['reference'],run['candidate']
        run['text_equal']=ref['text']==candidate['text']
        run['token_counts_equal']=ref['usage']['completion_tokens']==candidate['usage']['completion_tokens']
        run['decode_ratio']=candidate['observed_decode_tokens_per_second']/ref['observed_decode_tokens_per_second']
        save()
        assert run['token_counts_equal'] and (run['text_equal'] or a.allow_output_difference),run
    report['summaries'].append(dict(name=name,median_decode_change_percent=100*(statistics.median(r['decode_ratio'] for r in workload['runs'])-1),
        reference_tps=statistics.median(r['reference']['observed_decode_tokens_per_second'] for r in workload['runs']),
        candidate_tps=statistics.median(r['candidate']['observed_decode_tokens_per_second'] for r in workload['runs'])))
    save();print(report['summaries'][-1],flush=True)
# Passed denotes complete measurements and the requested parity checks, not a performance threshold.
report['passed']=True;save()
