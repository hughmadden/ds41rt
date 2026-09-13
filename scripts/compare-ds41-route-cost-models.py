#!/usr/bin/env python3
"""Compare row/unique and expert-group cost features on complete accepted-history traces."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output must be new')
    fitting = runpy.run_path(str(Path(__file__).with_name('fit-ds41-adaptive-policy.py')))
    sources = []
    for path in a.trace:
        observations, _ = fitting['observations'](path)
        rows = [x for x in observations if x['warm'] and not x['terminal'] and not x['missing']]
        if len(rows) < 100:
            p.error(f'insufficient complete warm rows: {path}')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows))
    models = {
        'rows_unique': ['intercept', 'rows', 'unique', 'extra_lane'],
        'rows_unique_extra_groups': ['intercept', 'rows', 'unique', 'extra_groups', 'extra_lane'],
    }

    def matrix(rows, features, history=False):
        result = []
        for r in rows:
            unique = r['history_unique' if history else 'unique']
            groups = r['history_groups' if history else 'groups']
            d = dict(intercept=1., rows=r['rows'], unique=unique,
                     extra_groups=groups-unique, extra_lane=r['lanes']-1)
            result.append([d[k] for k in features])
        return np.array(result)

    def fit(rows, features):
        X = matrix(rows, features)
        active = np.linalg.norm(X, axis=0) != 0
        beta = np.zeros(len(features))
        beta[active] = fitting['fit'](X[:, active], np.array([r['verify_us'] for r in rows]))
        return beta

    def error(rows, features, beta):
        y = np.array([r['verify_us'] for r in rows])
        result = dict(samples=len(rows))
        for name, history in [('actual_routes', False), ('accepted_history', True)]:
            e = (matrix(rows, features, history) @ beta - y) / y
            result[name] = dict(median_absolute_relative_error=float(np.median(abs(e))),
                               p90_absolute_relative_error=float(np.quantile(abs(e), .9)),
                               median_signed_relative_error=float(np.median(e)))
        return result

    report = dict(scope=__doc__, sources=[{k:v for k,v in s.items() if k!='rows'} for s in sources], models={})
    training, held = [], []
    for s in sources:
        split = [(i // 10) % 5 == 4 for i in range(len(s['rows']))]
        training.extend(r for r,h in zip(s['rows'],split) if not h)
        held.append([r for r,h in zip(s['rows'],split) if h])
    for name, features in models.items():
        beta = fit(training, features)
        result = dict(features=features, coefficients=beta.tolist(), training_samples=len(training),
                      block_heldout=[error(rows, features, beta) for rows in held], leave_trace_out=[])
        if len(sources)>1:
            for i,s in enumerate(sources):
                train=[r for j,other in enumerate(sources) if j!=i for r in other['rows']]
                b=fit(train,features)
                result['leave_trace_out'].append(dict(source=i,coefficients=b.tolist(),validation=error(s['rows'],features,b)))
        report['models'][name]=result
    report['limitations'] = ['Diagnostic model comparison, not a serving policy or counterfactual evaluation.',
        'Block holdout shares workload and request history; leave-trace-out is stricter but still limited to supplied traces.',
        'Changing row shapes, contexts and graph warmup remain confounders. No claim that expert groups fully explain C16 loss.']
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['models'],indent=2))


if __name__=='__main__':
    main()
