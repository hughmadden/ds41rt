"""CPU tests for the REAL compressor capture schema and weight resolution.

Every fixture is written in the exact schema emitted by ``input_trace.rs``
(48f0c712) so the loader cannot agree with a fictional layout.  No torch, no
CUDA, no GPU; only small synthetic CPU fixtures.  The actual read-only capture
is loaded when present, to prove the reproduced failure is fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compressor_replay_fakes as fakes  # noqa: E402,F401  (path helper only)

from compressor_replay import fixtures  # noqa: E402
from compressor_replay import schema  # noqa: E402

REAL_CAPTURE = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/"
    "flash-cap16-p2-compressor-diagnostic-01/activations"
)


def _root(tmp_path, geometry="pair", *, writer=True, seed=1, name="cap"):
    root = tmp_path / "activations"
    root.mkdir()
    directory = fixtures.write_synthetic_capture(
        root, name, geometry=geometry, weights_writer=writer, seed=seed
    )
    return root, directory


def _manifest(directory):
    return directory / "layer2-compressor-inputs.json"


def _rewrite(directory, mutator):
    path = _manifest(directory)
    data = json.loads(path.read_text())
    mutator(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def _patch(path: Path, offset: int, data: bytes):
    blob = bytearray(path.read_bytes())
    blob[offset:offset + len(data)] = data
    path.write_bytes(bytes(blob))


def _load(root, directory):
    return schema.load_capture(directory, activations_root=root)


# ---------------------------------------------------------------------------
# Real-schema fixtures: addressing, nonzero offsets, typed records.
# ---------------------------------------------------------------------------

def test_single_pending_row_nonzero_position(tmp_path):
    root, directory = _root(tmp_path, "single")
    capture = _load(root, directory)
    assert capture.rows == 1
    assert capture.slot_count == 4
    assert capture.device_descriptors == (0,)
    assert capture.p41_rows() == [0]
    wave_row = capture.wave_rows[0]
    assert wave_row["absolute_position"] == 41
    assert wave_row["predecessor"] == {"kind": "pending_slot", "slot": 0}
    # nonzero absolute-position offset: first token 40, logical compressed 20.
    assert wave_row["completed_latent"]["first_token"] == 40
    assert wave_row["completed_latent"]["logical_compressed_row"] == 20
    # The public JSON keys map into internal names; records are typed.
    assert set(capture.buffers) >= {
        "input", "projected", "scores", "output", "frequencies", "positions",
        "kv-values", "kv-scales", "pending-kv", "pending-scores",
        "descriptors-device",
    }
    assert capture.buffers["kv-values"].bytes == 256
    assert capture.buffers["kv-values"].dtype == "fp4e2m1"
    assert capture.buffers["output"].dtype == "bfloat16"


def test_pair_two_chunks_nonzero_prepared_offset(tmp_path):
    root, directory = _root(tmp_path, "pair")
    capture = _load(root, directory)
    assert capture.rows == 2
    assert [c["prepared_offset"] for c in capture.chunks] == [0, 1]
    assert capture.device_descriptors == (0, 1)
    assert [w["predecessor"]["kind"] for w in capture.wave_rows] == [
        "pending_slot", "pending_slot"
    ]
    assert capture.active_slots == (0, 1)
    # Each pending predecessor is the row's own lease slot.
    assert capture.row_map == ((0, 0), (1, 0))


def test_pair_earlier_wave_addressing(tmp_path):
    root, directory = _root(tmp_path, "pair_earlierwave")
    capture = _load(root, directory)
    assert capture.rows == 2
    assert len(capture.chunks) == 1
    assert capture.chunks[0]["tokens"] == 2
    assert capture.device_descriptors[0] == schema.DESCRIPTOR_SENTINEL
    assert capture.device_descriptors[1] == capture.slot_count
    assert capture.wave_rows[0]["predecessor"] == {"kind": "invalid_sentinel"}
    assert capture.wave_rows[0]["completed_latent"] is None
    assert capture.wave_rows[1]["predecessor"] == {
        "kind": "earlier_wave_row", "wave_row": 0
    }
    assert capture.wave_rows[1]["completed_latent"]["first_token"] == 40


def test_device_descriptor_mismatch_fails(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["wave_rows"][0].update(
        {"device_descriptor": 3}
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_even_position_carrying_predecessor_fails(tmp_path):
    root, directory = _root(tmp_path, "pair_earlierwave")
    # Make row 0 (even position 40) a pending-slot predecessor in both the
    # device bytes and the manifest record.
    _patch(directory / "layer2-compressor-descriptors-device.bin", 0,
           (0).to_bytes(8, "little"))
    _rewrite(directory, lambda d: (
        d["wave_rows"][0].update({"device_descriptor": 0,
                                  "predecessor": {"kind": "pending_slot",
                                                  "slot": 0}}),
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_missing_completed_latent_at_odd_position_fails(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["wave_rows"][0].update(
        {"completed_latent": None}
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_pending_slot_of_other_lease_fails(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: (
        d["chunks"][0]["lease"].update({"slot": 2}),
        d["wave_rows"][0].update({"predecessor": {"kind": "pending_slot",
                                                   "slot": 0}}),
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_host_descriptor_mismatch_is_preserved(tmp_path):
    root = tmp_path / "activations"
    root.mkdir()
    directory = fixtures.write_synthetic_capture(
        root, "cap", geometry="single", weights_writer=True, seed=3,
        device_matches_host=False,
    )
    capture = _load(root, directory)
    assert capture.device_matches_host is False
    assert capture.manifest_device_matches_host is False


# ---------------------------------------------------------------------------
# Malformed paths, shapes, byte extents.
# ---------------------------------------------------------------------------

def test_relative_traversal_path_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["buffers"]["input"].update(
        {"file": "../../evil.bin"}
    ))
    with pytest.raises(schema.PathEscapeError):
        _load(root, directory)


def test_absolute_path_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["buffers"]["input"].update(
        {"file": "/etc/passwd"}
    ))
    with pytest.raises(schema.PathEscapeError):
        _load(root, directory)


def test_wrong_shape_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["buffers"]["projected"].update(
        {"shape": [1, 511]}
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_bytecount_mismatch_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["buffers"]["scores"].update({"bytes": 4}))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_truncated_file_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    path = directory / "layer2-compressor-scores.bin"
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


def test_unknown_dtype_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["buffers"]["projected"].update(
        {"dtype": "float16"}
    ))
    with pytest.raises(schema.MalformedCaptureError):
        _load(root, directory)


# ---------------------------------------------------------------------------
# Weight resolution: shared files under projection-weights only.
# ---------------------------------------------------------------------------

def test_weights_resolved_from_projection_weights_not_recorded_remote(tmp_path):
    root, directory = _root(tmp_path, "single")
    capture = _load(root, directory)
    # The recorded remote directory is absolute and does not exist locally.
    assert capture.weights_provenance[
        "recorded_remote_directory_provenance_only"
    ] == "/trace/projection-weights"
    for role, ref in capture.weights.items():
        assert ref.path.parent == (root / "projection-weights")
    assert set(capture.weights) == {"wkv", "wgate", "norm"}
    assert capture.read_weight("norm") == (
        root / "projection-weights" / "layer2-compressor-norm-weight.bin"
    ).read_bytes()


def test_already_written_binds_to_first_writer(tmp_path):
    root = tmp_path / "activations"
    root.mkdir()
    first = fixtures.write_synthetic_capture(
        root, "first", geometry="single", weights_writer=True, seed=1
    )
    second = fixtures.write_synthetic_capture(
        root, "second", geometry="pair", weights_writer=False, seed=2
    )
    capture = _load(root, second)
    assert capture.weights_provenance["first_writer_manifest"].endswith(
        "first/layer2-compressor-inputs.json"
    )
    bindings = capture.weights_provenance["already_written_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["bound_to_first_writer"].endswith(
        "first/layer2-compressor-inputs.json"
    )
    assert first  # silence unused
    assert set(capture.weights) == {"wkv", "wgate", "norm"}


def test_missing_first_writer_manifest_rejected(tmp_path):
    root, directory = _root(tmp_path, "single", writer=False)
    with pytest.raises(schema.WeightError):
        _load(root, directory)


def test_exact_weight_tensor_name_required(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["weights"]["tensors"][0].update(
        {"tensor": "layers.2.attn.compressor.wkv.weight.extra"}
    ))
    with pytest.raises(schema.WeightError):
        _load(root, directory)


def test_duplicate_weight_role_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")

    def duplicate(data):
        first = dict(data["weights"]["tensors"][0])
        data["weights"]["tensors"].append(first)

    _rewrite(directory, duplicate)
    with pytest.raises(schema.WeightError):
        _load(root, directory)


def test_weight_file_size_mismatch_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    weight = root / "projection-weights" / "layer2-compressor-wkv-weight.bin"
    weight.write_bytes(weight.read_bytes()[:-2])
    with pytest.raises(schema.WeightError):
        _load(root, directory)


def test_traversal_weight_path_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["weights"]["tensors"][0].update(
        {"file": "../outside.bin"}
    ))
    with pytest.raises(schema.PathEscapeError):
        _load(root, directory)


def test_wrong_weight_dtype_rejected(tmp_path):
    root, directory = _root(tmp_path, "single")
    _rewrite(directory, lambda d: d["weights"]["tensors"][2].update(
        {"dtype": "float32"}
    ))
    with pytest.raises(schema.WeightError):
        _load(root, directory)


# ---------------------------------------------------------------------------
# The actual read-only capture (skipped when the fixture is not present).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CAPTURE.is_dir(),
                    reason="actual compressor capture not present")
def test_actual_capture_loads_three_manifests_and_five_p41_rows():
    captures = schema.load_captures(REAL_CAPTURE)
    assert len(captures) == 3
    assert sum(len(c.p41_rows()) for c in captures) == 5
    for capture in captures:
        assert capture.layer == 2
        assert capture.ratio == 2
        assert capture.device_matches_host is True
        assert set(capture.weights) == {"wkv", "wgate", "norm"}
        for ref in capture.weights.values():
            assert ref.path.parent == REAL_CAPTURE / "projection-weights"
    # The 2-row captures each hold two p41 pending rows; the 1-row capture one.
    assert sorted(len(c.p41_rows()) for c in captures) == [1, 2, 2]
