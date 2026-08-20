# Third-party source

All four dependency directories are pinned Git submodules. Initialize the
complete source graph after cloning DS4RT:

```bash
git submodule update --init --recursive
```

`sparkinfer/` is pinned from
<https://github.com/tpurtell/sparkinfer-glmrt>. DS4RT uses that source for
every Spark and coordinator CuTe AOT export; an independently installed
`b12x` or `sparkinfer` package is not a supported build input.

Initialize it after cloning DS4RT:

```bash
git submodule update --init --recursive third_party/sparkinfer
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json
```

The lock records both the fork commit and a deterministic content digest.
The digest keeps release archives verifiable after Git metadata is removed.
When intentionally updating the pin, update the submodule first, obtain the
new digest with `--print-tree-sha256`, update the lock, then run the full
verification command. The schema is:

```json
{
  "schema": 1,
  "repository": "https://github.com/tpurtell/sparkinfer-glmrt.git",
  "revision": "<lowercase 40-hex commit>",
  "source_tree_sha256": "<lowercase 64-hex digest>"
}
```

Generate the digest from a clean checkout; full verification also rejects a
wrong Git origin, a different `HEAD`, and tracked or non-ignored untracked
source changes. Never point a build at an unverified cache checkout.

`xgrammar/` is the pinned constrained-decoding implementation. It owns nested
source dependencies, which is why the top-level initialization command uses
`--recursive`.

```bash
python3 scripts/verify-xgrammar-source.py \
  --source third_party/xgrammar \
  --lock third_party/xgrammar.lock.json
```

`exllamav3/` is the separately pinned, official conversion source used to
produce DS4RT's calibrated expert-only EXL3 artifacts. It is a build-time tool,
not the serving runtime; SparkInfer consumes the emitted trellis tensors.

```bash
git submodule update --init third_party/exllamav3
python3 scripts/verify-exllamav3-source.py \
  --source third_party/exllamav3 \
  --lock third_party/exllamav3.lock.json
```

Its lock uses the same revision-plus-content-digest contract as SparkInfer so a
published quantization recipe remains reproducible after source archiving.

`gptqmodel/` is the pinned calibration and conversion engine used by the
reproducible DeepSeek V4 quantization workflow. It points at the user's fork so
DS4RT can carry source-decoding, routed-only inclusion, evidence, and resume
changes while they are qualified for upstreaming.

```bash
git submodule update --init third_party/gptqmodel
python3 scripts/verify-gptqmodel-source.py \
  --source third_party/gptqmodel \
  --lock third_party/gptqmodel.lock.json
```

The quantization container must run this verification before importing
GPTQModel. A Git checkout must be at the locked commit, have the expected fork
origin, and contain no source changes. Source archives are verified with the
same deterministic content digest even when Git metadata is absent.
