#!/usr/bin/env bash
set -euo pipefail

python_libdir="$(
  python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")'
)"
cute_libdir="$(
  python3 - <<'PY'
from pathlib import Path
import nvidia_cutlass_dsl

root = Path(next(iter(nvidia_cutlass_dsl.__path__))).resolve()
candidates = sorted(root.glob("cu*/lib"))
print(candidates[-1] if candidates else "")
PY
)"

runtime_libs=(/opt/ds4rt/lib)
if [[ -n "$python_libdir" ]]; then
  runtime_libs+=("$python_libdir")
fi
if [[ -n "$cute_libdir" ]]; then
  runtime_libs+=("$cute_libdir")
fi
runtime_path="$(IFS=:; printf '%s' "${runtime_libs[*]}")"
export LD_LIBRARY_PATH="$runtime_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DS4RT_PYTHON="${DS4RT_PYTHON:-$(command -v python3)}"

exec "$@"
