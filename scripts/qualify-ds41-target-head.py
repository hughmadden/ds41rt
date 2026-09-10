#!/usr/bin/env python3
"""Real target collapse/norm/vocabulary projection and selected-row handoffs."""
import argparse, ast, hashlib, json, struct
from pathlib import Path
from types import SimpleNamespace
import torch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'snapshot', 'vectors-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    source = a.reference_dir / 'inference/model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == lock['files']['inference/model.py']
    tree = ast.parse(source.read_text())
    ns = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'world_size': 1}
    norm = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RMSNorm'))
    exec(compile(ast.Module(body=[norm], type_ignores=[]), str(source), 'exec'), ns)
    for cls, method, target in [('Block', 'hc_pre', 'collapse'), ('ParallelHead', 'forward', 'project')]:
        node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls))
        fn = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method))
        fn.name = target
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((a.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    hashes = {}
    results = []

    def weight(name):
        path = a.snapshot / index[name]
        with path.open('rb') as f:
            if path not in headers:
                n = struct.unpack('<Q', f.read(8))[0]
                headers[path] = (n, json.loads(f.read(n)))
            n, h = headers[path]
            e = h[name]
            begin, end = e['data_offsets']
            f.seek(8 + n + begin)
            raw = f.read(end - begin)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(e['shape']).cuda()
    with torch.device('cuda'), torch.no_grad():
        gamma = weight('norm.weight')
        norm = ns['RMSNorm'](5120, 1e-20).to(dtype=torch.bfloat16)
        norm.weight.copy_(gamma)
        head = SimpleNamespace(weight=weight('head.weight').float())
        double_weight = head.weight.double()
        for case in json.loads((a.vectors_dir / 'head-cases.json').read_text()):
            rows = case['rows']
            prefix = case['prefix']
            t = {}
            payloads = {}
            for name, dtype, shape in [('residual', torch.bfloat16, (1, rows, 4, 5120)), ('pre', torch.float32, (1, rows, 4)), ('collapsed', torch.bfloat16, (1, rows, 5120)), ('normalized', torch.bfloat16, (1, rows, 5120)), ('logits', torch.float32, (1, rows, 129280))]:
                raw = (a.vectors_dir / f'{prefix}-{name}.bin').read_bytes()
                payloads[name] = hashlib.sha256(raw).hexdigest()
                t[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).cuda()
            collapsed = ns['collapse'](None, t['residual'], t['pre'])
            torch.testing.assert_close(t['collapsed'], collapsed, rtol=0.008, atol=0.002)
            normalized = norm(t['collapsed'])
            torch.testing.assert_close(t['normalized'], normalized, rtol=0.008, atol=0.002)
            logits = ns['project'](head, t['normalized'], True)
            torch.testing.assert_close(t['logits'], logits, rtol=2e-05, atol=2e-05)
            oracle = (t['normalized'].double() @ double_weight.T).float()
            torch.testing.assert_close(t['logits'], oracle, rtol=2e-05, atol=2e-05)
            chained = ns['project'](head, norm(collapsed), True)
            results.append(dict(**case, collapse_max_abs=(t['collapsed'].float() - collapsed.float()).abs().max().item(), norm_same_input_max_abs=(t['normalized'].float() - normalized.float()).abs().max().item(), fp64_oracle_max_abs=(t['logits'] - oracle).abs().max().item(), projection_same_input_max_abs=(t['logits'] - logits).abs().max().item(), chained_max_abs=(t['logits'] - chained).abs().max().item(), argmax_agreement=float((t['logits'].argmax(-1) == chained.argmax(-1)).float().mean().item()), payloads_sha256=payloads))
            print(f"PASS {prefix} rows={rows} real_target_head=true bound={case['bound']}", flush=True)
    a.output.write_text(json.dumps(dict(device=a.device, device_name=torch.cuda.get_device_name(a.device), torch_version=torch.__version__, cuda_version=torch.version.cuda, scope='Real target head stage comparisons only; no full-model logits claim', projection_tolerance=dict(rtol=2e-05, atol=2e-05), norm_tolerance=dict(rtol=0.008, atol=0.002), reference_sha256=lock['files']['inference/model.py'], weight_payloads_sha256=hashes, cases=results), indent=2) + '\n')
if __name__ == '__main__':
    main()
