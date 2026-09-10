#!/usr/bin/env python3
"""Qualify real backbone shared FP8 experts against the pinned reference kernels."""
import argparse, ast, hashlib, importlib.util, json, struct
from pathlib import Path
from types import SimpleNamespace
import torch
import tvm_ffi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'snapshot', 'vectors-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--capacities',type=int,nargs='+')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py', 'kernel.py'):
        assert hashlib.sha256((a.reference_dir / 'inference' / name).read_bytes()).hexdigest() == lock['files']['inference/' + name]
    source = a.reference_dir / 'inference/model.py'
    expert = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Expert')
    forward = next(n for n in expert.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    ns = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), ns)
    spec = importlib.util.spec_from_file_location('ds41_official_kernel', a.reference_dir / 'inference/kernel.py')
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    assert ref.tilelang.__version__ == '0.1.8'
    ref.act_quant_kernel.pass_configs = {**ref.act_quant_kernel.pass_configs, 'tir.disable_vectorize': True}
    torch.cuda.set_device(a.device); torch.set_default_dtype(torch.bfloat16)
    stream = torch.cuda.Stream()
    index = json.loads((a.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    headers, hashes, results = {}, {}, []
    def weight(name, dtype):
        path = a.snapshot / index[name]
        with path.open('rb') as f:
            if path not in headers:
                n = struct.unpack('<Q', f.read(8))[0]; headers[path] = (n, json.loads(f.read(n)))
            n, h = headers[path]; entry = h[name]; begin, end = entry['data_offsets']
            f.seek(8 + n + begin); raw = f.read(end - begin)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry['shape']).cuda()
    def linear(x, pair):
        q, s = ref.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        return ref.fp8_gemm(q, s, *pair, torch.float8_e8m0fnu, block_size=32)
    with torch.device('cuda'), torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
        for layer in (a.layers or range(40)):
            w = {name: (weight(f'layers.{layer}.ffn.shared_experts.{name}.weight', torch.float8_e4m3fn),
                        weight(f'layers.{layer}.ffn.shared_experts.{name}.scale', torch.float8_e8m0fnu)) for name in ('w1', 'w3', 'w2')}
            for rows in (a.capacities or ([1, 16, 80, 256, 1024, 4096] if layer in (0, 20, 39) else [80])):
                for case in range(2):
                    prefix = f'l{layer}-m{rows}-c{case}'; t, payloads = {}, {}
                    for name, width in [('input',5120), ('gate',2304), ('up',2304), ('intermediate',2304), ('output',5120)]:
                        raw = (a.vectors_dir / f'{prefix}-{name}.bin').read_bytes()
                        payloads[name] = hashlib.sha256(raw).hexdigest()
                        t[name] = torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(rows,width).cuda()
                    gate, up = linear(t['input'],w['w1']), linear(t['input'],w['w3'])
                    torch.testing.assert_close(t['gate'],gate,rtol=.008,atol=.002)
                    torch.testing.assert_close(t['up'],up,rtol=.008,atol=.002)
                    def activation(g,u):
                        return ns['forward'](SimpleNamespace(w1=lambda _:g,w3=lambda _:u,w2=lambda x:x,swiglu_limit=10.),t['input'])
                    local = activation(t['gate'],t['up'])
                    torch.testing.assert_close(t['intermediate'],local,rtol=0,atol=0)
                    down = linear(t['intermediate'],w['w2'])
                    torch.testing.assert_close(t['output'],down,rtol=.008,atol=.002)
                    expected_i = activation(gate,up)
                    chained = linear(expected_i,w['w2'])
                    # Bound FP8 quantization discontinuities from the two different
                    # BF16 intermediates, while retaining strict stage comparisons.
                    aq, asc = ref.act_quant(t['intermediate'],32,'ue8m0',torch.float8_e8m0fnu)
                    eq, esc = ref.act_quant(expected_i,32,'ue8m0',torch.float8_e8m0fnu)
                    delta = ((aq.float().reshape(rows,-1,32)*asc.float()[:,:,None])-
                             (eq.float().reshape(rows,-1,32)*esc.float()[:,:,None])).abs().reshape(rows,-1)
                    dw = w['w2'][0].float()*w['w2'][1].float().repeat_interleave(32,0).repeat_interleave(32,1)
                    propagated = delta @ dw.abs().T
                    bound = propagated + .004*(down.float().abs()+chained.float().abs()) + .008*down.float().abs()+.002
                    error = (t['output'].float()-chained.float()).abs()
                    assert bool((error <= bound).all())
                    results.append(dict(layer=layer,rows=rows,case=case,
                        gate_max_abs=(t['gate'].float()-gate.float()).abs().max().item(),
                        up_max_abs=(t['up'].float()-up.float()).abs().max().item(),
                        down_same_input_max_abs=(t['output'].float()-down.float()).abs().max().item(),
                        activation_exact=True,chained_max_abs=error.max().item(),
                        chained_rms=error.square().mean().sqrt().item(),
                        changed_intermediate_elements=int((t['intermediate']!=expected_i).sum()),
                        quantization_propagation_bound=True,payloads_sha256=payloads))
                    print(f'PASS {prefix} stages_close=true activation_exact=true propagation_bound=true',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,scope='Real backbone shared expert stages and bounded quantization propagation; excludes routed experts and full-model execution',
        reference_quantizer_compiler_overrides={'tir.disable_vectorize':True},weight_payloads_sha256=hashes,cases=results),indent=2)+'\n')


if __name__ == '__main__':
    main()
