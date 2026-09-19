#!/usr/bin/env bash
# Isolated WO-B arithmetic candidate. Not a complete serving native library.
set -euo pipefail
if [[ "$#" != 1 || "$1" != /* || -e "$1" ]]; then
  echo "usage: build-v41-wob-component.sh NEW_ABSOLUTE_OUTPUT_DIRECTORY" >&2
  exit 2
fi
if [[ ! "${DS41RT_SOURCE_REVISION:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DS41RT_SOURCE_REVISION must identify the committed source mount" >&2
  exit 2
fi
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="$1"
export PYTHONDONTWRITEBYTECODE=1
export DS41RT_SPARKINFER_SOURCE_DIR="${DS41RT_SPARKINFER_SOURCE_DIR:-/b12x}"
export DS41RT_SPARKINFER_LOCK_FILE="$source_root/third_party/sparkinfer.lock.json"
mkdir -- "$output"
python3 "$source_root/scripts/verify-sparkinfer-source.py" \
  --source "$DS41RT_SPARKINFER_SOURCE_DIR" --lock "$DS41RT_SPARKINFER_LOCK_FILE" \
  >"$output/source-check.txt"
python3 "$DS41RT_SPARKINFER_SOURCE_DIR/tests/gemm/test_native_aot_small_m.py" \
  >"$output/b12x-cpu-tests.txt" 2>&1
python3 "$source_root/python/tests/test_v41_fp8_aot_options.py" \
  >"$output/exporter-cpu-tests.txt" 2>&1
python3 - <<'PY' >"$output/toolchain.json"
import importlib.metadata
import json
import torch

props = torch.cuda.get_device_properties(0)
assert (props.major, props.minor, props.multi_processor_count) == (12, 0, 170)
print(json.dumps({'device': props.name, 'physical_sms': props.multi_processor_count,
    'capability': [props.major, props.minor], 'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'cutlass': importlib.metadata.version('nvidia-cutlass-dsl')}, indent=2))
PY
python3 "$source_root/python/tools/export_b12x_v41_fp8_aot.py" \
  --output-dir "$output/aot" --projections o_b \
  --rows 1,16,80,256,1024,4096 --wob-m16-split2
runtime_dir="$(python3 -m cutlass.cute.export.aot_config --libdir)"
objects=("$output/aot/"*.o)
if [[ ${#objects[@]} != 13 ]]; then
  echo "Expected twelve WO-B objects plus one HC project object" >&2
  exit 1
fi
nvcc --version >"$output/nvcc.txt"
nvcc -shared -std=c++17 -O3 -arch=sm_120 -Xcompiler=-fPIC \
  -I"$source_root/native/include" -I"$output/aot" \
  "$source_root/native/src/v41_fp8.cc" \
  "$source_root/native/cuda/kernels/v41_fp8.cu" "${objects[@]}" \
  -L"$runtime_dir" -lcute_dsl_runtime -lcuda \
  -Xlinker=-rpath -Xlinker="$runtime_dir" \
  -o "$output/libds41rt_wob_component.so"
python3 - "$output" "$DS41RT_SPARKINFER_LOCK_FILE" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import zoneinfo

output, lock = map(Path, sys.argv[1:])
manifest = json.loads((output / 'aot/v41_fp8.json').read_text())
assert manifest['physical_sms'] == 170 and manifest['capability'] == [12, 0]
assert manifest['wob_m16_split2'] is True and manifest['wob_m1_split1'] is False
assert len(manifest['variants']) == 6
assert all(v['label'].startswith('v41_o_b_') and
           v['split_k_slices'] == (2 if v['capacity'] in (1, 16) else 1)
           for v in manifest['variants'])
small = next(v for v in manifest['variants'] if v['capacity'] == 16)
assert small['wob_small_m_policy']['mma_tile_mn'] == [16, 64]
assert small['split_k_bytes'] == 2 * 16 * 5120 * 4
assert small['gemm_output_dtype'] == 'FP32'
files = [p for p in output.rglob('*') if p.is_file()]
record = {
    'kind': 'component-only-not-serving-native',
    'source_revision': os.environ['DS41RT_SOURCE_REVISION'],
    'b12x_lock': json.loads(lock.read_text()),
    'created_sydney': datetime.datetime.now(zoneinfo.ZoneInfo('Australia/Sydney')).isoformat(),
    'artifacts': {str(p.relative_to(output)): {'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)},
    'qualification': 'CPU contracts and compilation only; GPU oracle/performance pending',
}
(output / 'PROVENANCE.json').write_text(json.dumps(record, indent=2) + '\n')
PY
