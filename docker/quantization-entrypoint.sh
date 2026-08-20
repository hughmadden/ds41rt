#!/usr/bin/env bash
set -euo pipefail

report_path="${DS4RT_QUANT_PREFLIGHT_REPORT:-/tmp/ds4rt-quantization-preflight.json}"
# Keep command stdout machine-readable. The durable report remains at the
# requested path while the human-readable preflight copy goes to stderr.
python /opt/ds4rt/quantization/preflight.py --output "${report_path}" >&2

exec "$@"
