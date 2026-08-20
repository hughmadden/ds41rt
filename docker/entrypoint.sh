#!/usr/bin/env bash
set -euo pipefail

if [[ -d /workspace/ds4rt ]]; then
  export PATH="/workspace/ds4rt/scripts:$PATH"
  cd /workspace/ds4rt
fi
exec "$@"
