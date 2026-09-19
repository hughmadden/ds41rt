"""Actual-input sparse-attention component replay harness (CPU stubs + GPU runner).

Replays captured v41 sparse-attention inputs (see ``layer<N>-attention-inputs.json``
manifests) through the pinned native library, comparing BF16 output rows
byte-exactly against captured ``layer<N>-attention-values.bin`` rows.

Address/mask resolution is delegated to the reviewed integer oracle; capture
reading reuses the pinned extractor.  Nothing in this package imports torch at
module scope: the default CLI mode is validate/plan on CPU only, and the GPU
runner is imported lazily behind ``--execute``.
"""
