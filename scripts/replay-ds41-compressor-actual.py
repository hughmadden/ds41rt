#!/usr/bin/env python3
"""Component replay harness for actual v41 compressor producer captures.

Thin entry point: validates/plans on CPU by default; replays through the
pinned native library only with --execute.  See compressor_replay.cli.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compressor_replay.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
