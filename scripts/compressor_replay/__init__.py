"""Component replay harness for the v41 compressor producer capture.

Replays captured ratio-two compressor operands through the pinned native
library (``compressor_create/project/pool/destroy`` and
``compressed_kv_pack``) and byte-compares the downstream pipeline against the
capture.  Default operation is CPU-only validate/plan; GPU execution needs the
explicit ``--execute`` flag and a torch import that stays deferred to the
runner.

Nothing in this package changes production arithmetic or dispatch; it only
re-invokes the existing native exports against recorded device bytes.
"""
