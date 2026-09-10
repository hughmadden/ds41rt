#!/usr/bin/env bash
set -euo pipefail

if [[ -d /workspace/ds41rt ]]; then
  export PATH="/workspace/ds41rt/scripts:$PATH"
  cd /workspace/ds41rt
fi
exec "$@"
