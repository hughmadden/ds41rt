#!/usr/bin/env python3
"""Replay captured actual sparse-attention inputs through the native kernels.

Default mode validates the captures and prints the bounded replay plan on CPU
only; pass --execute (plus a fresh --output-dir) to run the cases on CUDA.
See scripts/sparse_replay/cli.py for the full contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_replay.cli import main

if __name__ == "__main__":
    sys.exit(main())
