#!/usr/bin/env python3
"""Cross-check raw conditional confidence on two explicitly identified request cohorts.

Use a fixed full-draft trace to avoid the adaptive policy censoring low-confidence
suffixes. A conditional label is observed only while all earlier tokens match.
Cohorts are initial request IDs versus subsequent IDs, not inferred workloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--initial-requests', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.initial_requests < 1:
        p.error('output must be new and initial request count positive')
    labels, decisions = [], []
    for raw in args.trace.read_text().splitlines():
        if 'native draft policy observation' not in raw:
            continue
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        f = dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', line.strip()))
        if int(f['generated']) <= 6 or any(f[k] == 'true' for k in ['eos', 'length_limit', 'constrained']):
            continue
        d = {k: int(f[k]) for k in ['request_id', 'verifier_rows', 'matched_prefix']}
        d['confidence'] = json.loads(f['raw_confidence'])
        assert 0 <= d['matched_prefix'] < d['verifier_rows'] <= 6
        decisions.append(d)
        for j in range(min(d['verifier_rows']-1, d['matched_prefix']+1)):
            labels.append([d['request_id'], j, d['confidence'][j], int(d['matched_prefix'] > j)])
    c = np.array(labels)
    if len(c) == 0:
        p.error('no warm uncensored conditional labels')
    X = np.column_stack([np.ones(len(c)), c[:, 2]])
    y = c[:, 3]
    report = dict(scope=__doc__, trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  initial_requests=args.initial_requests, conditional_labels=len(c), cohorts=[])
    for initial in [True, False]:
        held = (c[:, 0] <= args.initial_requests) == initial
        if not held.any() or held.all():
            p.error('both request cohorts need observed labels')
        beta = np.array([0., 1.])
        for _ in range(30):
            probability = sigmoid(X[~held] @ beta)
            w = probability*(1-probability)
            beta -= np.linalg.solve((X[~held].T*w) @ X[~held]+.1*np.eye(2),
                                    X[~held].T @ (probability-y[~held])+.1*beta)
        raw, fitted = sigmoid(c[held, 2]), sigmoid(X[held] @ beta)
        ds = [d for d in decisions if (d['request_id'] <= args.initial_requests) == initial]
        positions = []
        for j in range(5):
            mask = held & (c[:, 1] == j)
            if mask.any():
                positions.append(dict(position=j+1, samples=int(mask.sum()),
                    observed_acceptance=float(np.mean(y[mask])), raw_mean_probability=float(np.mean(sigmoid(c[mask, 2])))))
        expected = lambda d, b: 1+np.cumprod(sigmoid(b[0]+b[1]*np.array(d['confidence'][:d['verifier_rows']-1]))).sum()
        report['cohorts'].append(dict(initial=initial, labels=int(held.sum()), decisions=len(ds),
            observed_acceptance=float(np.mean(y[held])), raw_mean_probability=float(np.mean(raw)),
            transform_trained_on_other_cohort=beta.tolist(),
            raw_brier=float(np.mean((raw-y[held])**2)), crossfit_brier=float(np.mean((fitted-y[held])**2)),
            mean_emitted_raw=float(np.mean([expected(d, [0., 1.]) for d in ds])),
            mean_emitted_crossfit=float(np.mean([expected(d, beta) for d in ds])),
            mean_emitted_observed=float(np.mean([1+d['matched_prefix'] for d in ds])), positions=positions))
    report['limitations'] = ['Same prompt corpus across cohorts; not independent-workload validation.',
        'Cohorts include each request through its drain, not only rounds at initial concurrency.',
        'Mean calibration does not establish per-decision counterfactual accuracy or justify coefficient promotion.']
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['cohorts'], indent=2))

if __name__ == '__main__':
    main()
