"""Build host selection must not contact unavailable serving ranks."""
import subprocess
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "release-common.sh"


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("", ["ostrich", "dodo", "emu", "kiwi"]),
        ("ostrich,dodo", ["ostrich", "dodo"]),
        ("dodo", ["dodo"]),
        ("emu,emu", None),
        ("elsewhere", None),
        ("ostrich,", None),
        (",dodo", None),
        ("ostrich,,dodo", None),
    ],
)
def test_selected_build_hosts(requested, expected):
    result = subprocess.run(
        [
            "bash", "-euc",
            'source "$1"; SPARK_0_HOST=ostrich; SPARK_1_HOST=dodo; '
            'SPARK_2_HOST=emu; SPARK_3_HOST=kiwi; '
            'release_select_build_hosts "$2"; '
            'printf "%s\\n" "${RELEASE_BUILD_HOSTS[@]}"',
            "bash", str(COMMON), requested,
        ],
        capture_output=True, text=True,
    )
    if expected is None:
        assert result.returncode == 2
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == expected
