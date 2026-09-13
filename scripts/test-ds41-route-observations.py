#!/usr/bin/env python3
"""Regression tests for accepted-only route history in offline cost fitting."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

observe = runpy.run_path(str(Path(__file__).with_name('fit-ds41-adaptive-policy.py')))['observations']


def round_lines(generated, start, rows, accepted, ids):
    lines = [f'native route policy observation layer={layer} owners={[(1,p) for p in range(start,start+rows)]} route_ids={json.dumps(ids)}' for layer in range(40)]
    lines.append(f'native draft policy observation request_id=1 lane=0 context_tokens={start} verifier_rows={rows} accepted_inputs={accepted} matched_prefix=1 generated={generated} raw_confidence=[0,0,0,0,0] eos=false length_limit=false constrained=false')
    lines.append('native scheduler round verify_us=1000 draft_us=100 requests=1')
    return lines


def parse(lines):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'trace.log'
        path.write_text('\n'.join(lines) + '\n')
        return observe(path)[0]


class AcceptedRoutes(unittest.TestCase):
    def test_rejected_suffix_cannot_enter_history(self):
        rows = parse(round_lines(1, 0, 6, 2, [0]*6+[1]*6+[99]*24) +
                     round_lines(3, 2, 2, 2, [1]*12))
        self.assertEqual((rows[1]['history_unique'], rows[1]['history_groups']), (2, 2))
        self.assertEqual((rows[1]['unique'], rows[1]['groups']), (1, 1))
        self.assertEqual(rows[1]['missing'], 0)

    def test_insufficient_accepted_history_is_not_qualified(self):
        rows = parse(round_lines(1, 0, 6, 2, [0]*36) +
                     round_lines(3, 2, 6, 2, [1]*36))
        self.assertEqual(rows[1]['missing'], 40)
        self.assertEqual(rows[1]['groups'], 3)

    def test_duplicate_layer_cannot_double_group_count(self):
        lines = round_lines(1, 0, 6, 2, [0]*36)
        with self.assertRaisesRegex(ValueError, 'duplicate or malformed'):
            parse([lines[0]] + lines)


if __name__ == '__main__':
    unittest.main()
