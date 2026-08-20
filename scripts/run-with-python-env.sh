#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python="${DS4RT_PYTHON:-}"
if [ -z "$python" ] && [ -x "$repo_root/.venv/bin/python" ]; then
  python="$repo_root/.venv/bin/python"
fi
if [ -z "$python" ]; then
  python="$(command -v python3 || true)"
fi
if [ -z "$python" ] || [ ! -x "$python" ]; then
  echo "Python interpreter not found; set DS4RT_PYTHON" >&2
  exit 2
fi

python="$(realpath -s "$python")"
python_config="$("$python" - <<'PY'
import os
import sys
import sysconfig

print(sys.base_prefix)
print(sysconfig.get_config_var("LIBDIR") or "")
print(os.pathsep.join(
    path for path in sys.path if path and "site-packages" in path
))
PY
)"
python_home="$(sed -n '1p' <<<"$python_config")"
python_libdir="$(sed -n '2p' <<<"$python_config")"
python_module_path="$(sed -n '3p' <<<"$python_config")"

export DS4RT_PYTHON="$python"
export PYO3_PYTHON="${PYO3_PYTHON:-$python}"
export PYTHONHOME="${PYTHONHOME:-$python_home}"
export PATH="$(dirname "$python"):$PATH"
if [ -n "$python_libdir" ]; then
  export LD_LIBRARY_PATH="$python_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python_paths=()
if [ -d "$repo_root/python/reference" ]; then
  python_paths+=("$repo_root/python/reference")
fi
if [ -d "$repo_root/third_party/sparkinfer" ]; then
  python_paths+=("$repo_root/third_party/sparkinfer")
fi
if [ -n "$python_module_path" ]; then
  python_paths+=("$python_module_path")
fi
if [ -n "${PYTHONPATH:-}" ]; then
  python_paths+=("$PYTHONPATH")
fi
if [ "${#python_paths[@]}" -gt 0 ]; then
  export PYTHONPATH="$(IFS=:; printf '%s' "${python_paths[*]}")"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/run-with-python-env.sh COMMAND [ARG ...]" >&2
  exit 2
fi
exec "$@"
