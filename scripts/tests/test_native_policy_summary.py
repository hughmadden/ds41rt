import runpy
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).parents[1] / 'summarize-ds41-native-policy.py'))


def test_conditional_acceptance_censors_rejected_history_and_terminal_rows():
    base = dict(request_id=1, lane=0, generated=1, rows=6, confidence=[0.] * 5,
                constrained=False, terminal=False)
    rows = [dict(base, matched=1), dict(base, matched=5),
            dict(base, matched=0, terminal=True), dict(base, matched=0, constrained=True)]
    result = MODULE['summarize'](rows, [])['calibration']
    assert [r['samples'] for r in result] == [2, 2, 1, 1, 1]
    assert result[0]['bins'][0]['observed_acceptance'] == 1
    assert result[1]['bins'][0]['observed_acceptance'] == .5
    assert result[2]['bins'][0]['observed_acceptance'] == 1


def test_short_full_match_does_not_label_unverified_suffix():
    row = dict(request_id=1, lane=1, generated=1, rows=2, confidence=[0.] * 5,
               constrained=False, terminal=False, matched=1)
    result = MODULE['summarize']([row], [])['calibration']
    assert [r['samples'] for r in result] == [1, 0, 0, 0, 0]
