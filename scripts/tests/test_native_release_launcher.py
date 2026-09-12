from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class NativeReleaseLauncherTest(unittest.TestCase):
    def test_shell_is_valid_and_help_exposes_native_controls(self) -> None:
        subprocess.run(
            ["bash", "-n", "run.sh", "scripts/release-common.sh"],
            cwd=ROOT,
            check=True,
        )
        help_text = subprocess.run(
            ["./run.sh", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for option in (
            "--listen",
            "--concurrency",
            "--kv-pool-size",
            "--memory-reservation",
            "--prefix-cache-entries",
            "--max-context-tokens",
            "--max-output-tokens",
            "--prefill-batch-tokens",
            "--dspark",
            "--no-dspark",
        ):
            self.assertIn(option, help_text)

    def test_first_release_defaults_are_native(self) -> None:
        script = r'''
source scripts/release-common.sh
release_load_config ds41rt.config
printf '%s\n' "$MODEL_ID" "$MODEL_REVISION" "$EXPERT_FORMAT" "$SPARKINFER_EXL3" \
  "$CONCURRENCY" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" \
  "$MAX_OUTPUT_TOKENS" "$ADDR" "$EXPERT_PORT"
'''
        values = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(
            values,
            [
                "deepseek-ai/DeepSeek-V4.1-Flash",
                "dba1be0a40aa45a94ad051997016db3960a90277",
                "native",
                "disable",
                "16",
                "24",
                "1048576",
                "393216",
                "0.0.0.0:8000",
                "19441",
            ],
        )

    def test_launchers_use_ds41rt_container_names(self) -> None:
        combined = (ROOT / "build.sh").read_text() + (ROOT / "run.sh").read_text()
        common = (ROOT / "scripts/release-common.sh").read_text()
        self.assertNotIn("ds4rt", combined.lower())
        self.assertIn("ds41rt-coordinator", common)
        self.assertIn("ds41rt-spark-expert", common)
        self.assertIn("expertd-native", combined)
        self.assertIn("serve-native", combined)

    def test_invalid_direct_overrides_fail_before_external_checks(self) -> None:
        for args, message in (
            (["--concurrency", "17"], "CONCURRENCY must be in 1..16"),
            (["--prefix-cache-entries", "129"], "PREFIX_CACHE_ENTRIES must be in 0..128"),
            (["--max-context-tokens", "1048577"], "MAX_CONTEXT_TOKENS must be in 1..1048576"),
            (["--max-output-tokens", "393217"], "MAX_OUTPUT_TOKENS must be in 1..393216"),
        ):
            result = subprocess.run(
                ["./run.sh", *args], cwd=ROOT, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
