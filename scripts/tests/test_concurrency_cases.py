"""Exercise release concurrency case selection against a local SSE fixture."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
CODE = '''```python
# response {number}
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping intervals in sorted order."""
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged
assert merge_intervals([]) == []
assert merge_intervals([[1, 3], [2, 4]]) == [[1, 4]]
assert merge_intervals([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]
```'''


class ConcurrencyCases(unittest.TestCase):
    def run_fixture(self, case, invalid=False):
        requests = []
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    number = len(requests)
                    requests.append(body)
                if case == 'counting':
                    text = ', '.join(str(n) for n in range(1, 201))
                elif case == 'code':
                    text = 'invalid code' if invalid else CODE.format(number=number)
                else:
                    text = f'Virtual memory explanation variant {number}.'
                events = [
                    {'choices': [{'delta': {'content': text}}]},
                    {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                    {'choices': [], 'usage': {'prompt_tokens': 32, 'completion_tokens': 17,
                     'total_tokens': 49, 'prompt_cache_hit_tokens': 0 if number == 0 else 32}},
                ]
                payload = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(payload.encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'result.json'
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/bench-ds41-concurrent-api.py'),
                    '--base-url', f'http://127.0.0.1:{server.server_port}', '--case', case,
                    '--concurrency', '1', '2', '--repeats', '1', '--nonce', 'fixture',
                    '--output', str(output)], capture_output=True, text=True)
                report = json.loads(output.read_text()) if output.exists() else None
                return result, report, requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cases_keep_warm_accounting_and_allow_prose_and_code_variation(self):
        for case, limit in [('counting', 640), ('code', 320), ('topic', 384)]:
            with self.subTest(case=case):
                result, report, requests = self.run_fixture(case)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(report['passed'])
                self.assertEqual(report['case'], case)
                self.assertEqual(len(requests), 4)
                self.assertTrue(all(r['max_tokens'] == limit for r in requests))
                self.assertTrue(all(r['thinking'] == {'type': 'disabled'} for r in requests))
                rows = [r for batch in report['records'] for r in batch['rows']]
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(r['result']['usage']['prompt_cache_hit_tokens'] == 32 for r in rows))
                texts = {r['result']['text'] for r in rows}
                self.assertEqual(len(texts), 1 if case == 'counting' else 3)
                self.assertTrue(all(r['output_checks']['prose_quality_assessed'] is False for r in rows))
                if case == 'topic':
                    self.assertTrue(all(r['output_checks']['objective_checks_passed'] is None for r in rows))

    def test_invalid_code_is_not_reported_as_a_pass(self):
        result, report, _ = self.run_fixture('code', invalid=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report and report.get('passed'))


if __name__ == '__main__':
    unittest.main()
