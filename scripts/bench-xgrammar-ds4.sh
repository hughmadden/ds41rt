#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="557becfb64c503ae9c04344b0047661f43f44320"
SOURCE_DIR="${DS41RT_XGRAMMAR_SOURCE_DIR:-$ROOT/third_party/xgrammar}"
LOCK_FILE="${DS41RT_XGRAMMAR_LOCK_FILE:-$ROOT/third_party/xgrammar.lock.json}"
BUILD_DIR="${DS41RT_XGRAMMAR_BUILD_DIR:-/tmp/ds41rt-xgrammar-v0.2.3-build}"
FLASH_SNAPSHOT="${DS41RT_FLASH_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
TOKENIZER="${DS41RT_XGRAMMAR_TOKENIZER:-$FLASH_SNAPSHOT/tokenizer.json}"
FIXTURE="$ROOT/native/tools/fixtures/ds4_required_tool_calls.json"
BENCH_SOURCE="$ROOT/native/tools/xgrammar_ds4_bench.cc"
BENCH_BINARY="$BUILD_DIR/xgrammar_ds4_bench"

python3 "$ROOT/scripts/verify-xgrammar-source.py" \
  --source "$SOURCE_DIR" \
  --lock "$LOCK_FILE"
ACTUAL_COMMIT="$EXPECTED_COMMIT"

test -s "$TOKENIZER"
mkdir -p "$BUILD_DIR"
printf '%s\n' \
  'set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)' \
  'set(XGRAMMAR_BUILD_CXX_TESTS OFF)' \
  >"$BUILD_DIR/config.cmake"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target xgrammar -j "${DS41RT_BUILD_JOBS:-8}" >/dev/null

"${CXX:-c++}" \
  -O3 -DNDEBUG -std=c++17 -flto=auto -Wall -Wextra -Werror \
  -I"$SOURCE_DIR/include" \
  -I"$SOURCE_DIR/3rdparty/picojson" \
  -I"$SOURCE_DIR/3rdparty/dlpack/include" \
  "$BENCH_SOURCE" "$BUILD_DIR/libxgrammar.a" \
  -pthread -o "$BENCH_BINARY"

exec "$BENCH_BINARY" \
  --tokenizer "$TOKENIZER" \
  --fixture "$FIXTURE" \
  --xgrammar-commit "$ACTUAL_COMMIT" \
  "$@"
