"""Exercise the actual exporter with stub compiler/device; no GPU imports."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools/export_b12x_v41_fp8_aot.py"


class V41Fp8AotOptionsTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("fp8_export_options_test", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"_pinned_sparkinfer": types.ModuleType("_pinned_sparkinfer")}):
            spec.loader.exec_module(self.module)
        self.calls = []
        self.ignore_override = False

        class Compiled:
            def export_to_c(self, directory, label, symbol):
                for suffix in (".h", ".o"):
                    (Path(directory) / (label + suffix)).write_bytes(label.encode())

        def gemm(**kwargs):
            self.calls.append(kwargs)
            slices = 2 if (kwargs["size_m"] == 1 and kwargs["size_n"] >= 4096
                and kwargs["size_k"] >= 4096 and kwargs["num_groups"] == 1) else 1
            if kwargs.get("force_split_k_one") and not self.ignore_override:
                slices = 1
            return Compiled(), slices

        def module(name, **attrs):
            value = types.ModuleType(name)
            value.__dict__.update(attrs)
            return value

        props = types.SimpleNamespace(major=12, minor=0, multi_processor_count=170, name="stub")
        self.modules = {
            "torch": module("torch", bfloat16="BF16", device=lambda *args: "stub:0",
                cuda=types.SimpleNamespace(init=lambda: None, current_device=lambda: 0,
                    get_device_properties=lambda device: props)),
            "b12x._lib.dense_gemm": module("b12x._lib.dense_gemm", compile_dense_gemm_mxfp8_aot=gemm),
            "b12x._lib.quant.mxfp8_rows": module("b12x._lib.quant.mxfp8_rows",
                compile_mxfp8_rows_quant_aot=lambda **kwargs: Compiled()),
            "b12x.gemm.wo_projection._quant_cute": module("b12x.gemm.wo_projection._quant_cute",
                compile_wo_grouped_quant_aot=lambda **kwargs: Compiled()),
            "b12x.gemm._shared.block_fp8": module("b12x.gemm._shared.block_fp8",
                _block_fp8_linear_scratch_layout=lambda **kwargs: types.SimpleNamespace(
                    nbytes=4096, x_values_offset_bytes=0, x_scale_rows_offset_bytes=1024,
                    x_scale_mma_offset_bytes=2048, x_scale_mma_physical_shape=(32, 4))),
            "b12x.norm.mhc._v41_project": module("b12x.norm.mhc._v41_project",
                compile_v41_mhc_project_aot=lambda: Compiled()),
        }
        self.module.validate_abi = lambda *args: {"stub": True}
        self.module.dispatch_header = lambda output, manifest: (output / "v41_fp8_variants.h").write_text("stub")

    def export(self, output, **kwargs):
        with patch.dict(sys.modules, self.modules), patch.dict(self.module.os.environ, {}), contextlib.redirect_stdout(io.StringIO()):
            self.module.export(output, (1, 16), **kwargs)
        return json.loads((output / "v41_fp8.json").read_text())

    def test_default_preserves_every_projection_compiler_call(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.export(Path(directory) / "default")
            before = list(self.calls)
            self.calls.clear()
            second = self.export(Path(directory) / "explicit", wob_m1_split1=False)
        self.assertEqual(before, self.calls)
        self.assertEqual(first, second)
        self.assertFalse(first["wob_m1_split1"])
        self.assertTrue(all("force_split_k_one" not in call for call in before))

    def test_opt_in_changes_only_wo_b_capacity1_and_records_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            default = self.export(Path(directory) / "default")
            before = list(self.calls)
            self.calls.clear()
            explicit = self.export(Path(directory) / "split1", wob_m1_split1=True)
        changed = []
        for old, new in zip(before, self.calls, strict=True):
            if old != new:
                changed.append((new["size_m"], new["size_k"], new["size_n"]))
                self.assertEqual(new, dict(old, force_split_k_one=True))
        self.assertEqual(changed, [(1, 8192, 5120)])
        self.assertTrue(explicit["wob_m1_split1"])
        for old, new in zip(default["variants"], explicit["variants"], strict=True):
            if new["label"] == "v41_o_b_fp8_m1":
                self.assertEqual(new["split_k_slices"], 1)
                self.assertEqual(new["split_k_bytes"], 0)
                self.assertEqual(new["gemm_output_dtype"], "BF16")
                self.assertTrue(new["force_split_k_one"])
            else:
                self.assertEqual(old, new)

    def test_compiler_ignoring_override_cannot_publish_manifest(self):
        self.ignore_override = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad"
            with self.assertRaisesRegex(ValueError, "split1"):
                self.export(output, wob_m1_split1=True)
            self.assertFalse((output / "v41_fp8.json").exists())

    def test_requested_profile_requires_target_projection_and_capacity(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, self.modules):
            for rows, projections in [((16,), self.module.PROJECTIONS),
                ((1,), tuple(p for p in self.module.PROJECTIONS if p[0] != "o_b"))]:
                with self.assertRaisesRegex(ValueError, "o_b.*capacity1"):
                    self.module.export(Path(directory), rows, projections, wob_m1_split1=True)
        self.assertEqual(self.calls, [])

    def test_cli_forwards_explicit_option(self):
        seen = []
        self.module.export = lambda *args, **kwargs: seen.append((args, kwargs))
        with patch.object(sys, "argv", [str(SCRIPT), "--output-dir", "/unused", "--wob-m1-split1"]):
            self.module.main()
        self.assertTrue(seen[0][1]["wob_m1_split1"])


if __name__ == "__main__":
    unittest.main()
