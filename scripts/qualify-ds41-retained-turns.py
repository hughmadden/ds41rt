#!/usr/bin/env python3
"""Verify N completed turns survive alongside N prompt snapshots, then LRU eviction.

Use an exclusive native endpoint with scheduler DEBUG logging. Long followups
are cancelled after first content so inspecting a retained turn does not insert
another completed turn. Admission logs prove the exact committed cache frontier.
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', required=True)
parser.add_argument('--container', required=True)
parser.add_argument('--turns', type=int, default=24)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--collect-from', type=Path, help='Validate preserved requests and admission logs without rerunning inference.')
args = parser.parse_args()
assert 2 <= args.turns <= 128
assert not args.output.exists(), 'preserve existing evidence; choose a new output path'
records = dict(turn_limit=args.turns, cases=[], passed=False)

def save():
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')

def logs():
    result = subprocess.run(['docker','logs',args.container],capture_output=True,text=True,check=True)
    return re.sub(r'\x1b\[[0-9;]*m','',result.stdout+result.stderr)

pattern = re.compile(r'native prefix admission request_id=(\d+) prompt_tokens=(\d+) cached_tokens=(\d+)')

def completed_frontiers(response):
    # Target decoding leaves the last emitted token as an unevaluated anchor.
    # verify_dspark_greedy also commits EOS when it matched a draft input;
    # a correction/bonus EOS remains unevaluated. Both retain every content token.
    usage = response['usage']
    frontiers = [usage['total_tokens'] - 1]
    fingerprint = response['system_fingerprint']
    assert fingerprint in (
        'ds41rt-native-fp4-kv', 'ds41rt-native-fp4-kv-dspark',
        # Preserve replay support for the pre-migration baseline evidence.
        'ds41rt-native-fp8-kv', 'ds41rt-native-fp8-kv-dspark',
    )
    if fingerprint.endswith('-dspark'):
        frontiers.append(usage['total_tokens'])
    assert min(frontiers) > usage['prompt_tokens']
    return frontiers

def validate():
    cases={case['name']:case for case in records['cases']}
    count=records['turn_limit']
    assert len(cases)==len(records['cases'])==3*count+3
    admissions={int(i):(int(p),int(c)) for i,p,c in pattern.findall(records['scheduler_log'])}
    expected={}
    for i in range(count):
        seed=cases[f'seed-{i}']['response']
        repeat=cases[f'prompt-repeat-{i}']['response']
        assert seed['choices'][0]['message']['content'].strip()=='OK'
        assert seed['choices'][0]['finish_reason']=='stop'
        assert repeat['choices'][0]['message']==seed['choices'][0]['message']
        assert repeat['usage']['prompt_cache_hit_tokens']==repeat['usage']['prompt_tokens']
        expected[f'turn-resume-{i}']=completed_frontiers(seed)
    expected['oldest-turn-evicted']=[0]
    expected['next-turn-retained']=completed_frontiers(cases['seed-1']['response'])
    for name,frontiers in expected.items():
        case=cases[name]
        assert case['cancelled_after_content']
        actual=admissions[case['request_id']]
        case['admission']=dict(prompt_tokens=actual[0],cached_tokens=actual[1])
        case.pop('expected_cached', None)  # Replace the original target-only assumption.
        case['expected_committed_frontiers']=frontiers
        save()
        assert actual[1] in frontiers,(name,actual,frontiers)
    # The untouched second turn must preserve exactly the frontier seen before
    # insertion of turn N+1, including whether its EOS was committed.
    assert cases['next-turn-retained']['admission']['cached_tokens'] == cases['turn-resume-1']['admission']['cached_tokens']
    records['passed']=True
    save()
    print(f'PASS: {count} complete prompt hits, {count} exact completed-turn resumptions, oldest-turn eviction, newer-turn survival',flush=True)

if args.collect_from:
    records=json.loads(args.collect_from.read_text())
    records['collected_from']=str(args.collect_from)
    validate()
    raise SystemExit(0)

initial = pattern.findall(logs())
start_id = max([int(row[0]) for row in initial],default=0)

def call(name, body, cancel=False):
    record = dict(name=name,request=body,cancel=cancel,request_id=start_id+len(records['cases'])+1)
    records['cases'].append(record)
    save()
    request = urllib.request.Request(args.base_url+'/v1/chat/completions', data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    started=time.monotonic()
    with urllib.request.urlopen(request,timeout=180) as response:
        if cancel:
            events=[]
            record['events']=events
            for line in response:
                if not line.startswith(b'data: '):continue
                data=line[6:].strip()
                assert data != b'[DONE]', 'followup finished before cancellation'
                event=json.loads(data);events.append(event)
                if any(c.get('delta',{}).get('content') for c in event.get('choices',[])):
                    record['cancelled_after_content']=True
                    break
            assert record.get('cancelled_after_content')
        else:
            record['response']=json.load(response)
    record['elapsed_seconds']=time.monotonic()-started
    save()
    return record

session=uuid.uuid4().hex
seeds=[]
for i in range(args.turns):
    body={'model':'deepseek-ai/DeepSeek-V4.1-Flash','messages':[{'role':'user','content':f'Session {session}, item {i}. Reply only OK.'}],
          'thinking':{'type':'disabled'},'temperature':0,'max_tokens':16}
    seed=call(f'seed-{i}',body)
    assert seed['response']['choices'][0]['message']['content'].strip()=='OK'
    assert seed['response']['choices'][0]['finish_reason']=='stop'
    seeds.append(seed)
# Same completed frontiers replace existing values; they do not consume new turns.
for i,seed in enumerate(seeds):
    repeated=call(f'prompt-repeat-{i}',seed['request'])
    usage=repeated['response']['usage']
    assert usage['prompt_cache_hit_tokens']==usage['prompt_tokens']
    assert repeated['response']['choices'][0]['message']==seed['response']['choices'][0]['message']

def followup(seed, marker):
    return dict(seed['request'],stream=True,max_tokens=2048,messages=seed['request']['messages']+[
        {'role':'assistant','content':seed['response']['choices'][0]['message']['content']},
        {'role':'user','content':f'{marker}. Count from 1 to 10000, separated by commas. Output only the numbers and continue without stopping early.'}])

for i,seed in enumerate(seeds):
    case=call(f'turn-resume-{i}',followup(seed,'Resume check'),cancel=True)
    case['expected_committed_frontiers']=completed_frontiers(seed['response'])
    save()
# The resumptions touched turns in order; the oldest should now be seed 0.
body=dict(seeds[0]['request'],messages=[{'role':'user','content':f'New session {session}. Reply only OK.'}])
call('new-turn-over-limit',body)
case=call('oldest-turn-evicted',followup(seeds[0],'Eviction check'),cancel=True)
case['expected_committed_frontiers']=[0]
case=call('next-turn-retained',followup(seeds[1],'Survival check'),cancel=True)
case['expected_committed_frontiers']=completed_frontiers(seeds[1]['response'])
text=logs();records['scheduler_log']=text
save()
validate()
