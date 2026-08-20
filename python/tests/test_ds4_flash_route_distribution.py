from __future__ import annotations

from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import collect_ds4_flash_route_distribution as routes  # noqa: E402
from collect_expert_route_bank import RouteFragment  # noqa: E402


def fragment(layer_id: int, rows: int) -> RouteFragment:
    return RouteFragment(
        request_id_base=layer_id * 1_000,
        layer_id=layer_id,
        physical_m=rows,
        source_kinds=("PrefillChunk",),
        routes=tuple(tuple(range(routes.TOP_K)) for _ in range(rows)),
    )


def test_validate_fragments_covers_base_and_optional_mtp_layers() -> None:
    fragments = [fragment(layer_id, 2) for layer_id in range(routes.BASE_LAYERS)]
    fragments.extend(fragment(layer_id, 1) for layer_id in range(43, 46))

    counts, rows, cohorts = routes.validate_fragments(
        fragments,
        {
            "request_expert_batch_rows": routes.BASE_LAYERS,
            "request_expert_batch_routes": routes.BASE_LAYERS * routes.TOP_K,
        },
    )

    assert rows[:43] == [2] * 43
    assert rows[43:] == [1] * 3
    assert counts[0][:6] == [2] * 6
    assert counts[43][:6] == [1] * 6
    assert cohorts == {"PrefillChunk": 89}


def test_validate_fragments_rejects_unequal_base_layer_rows() -> None:
    fragments = [fragment(layer_id, 2) for layer_id in range(routes.BASE_LAYERS)]
    fragments[7] = fragment(7, 1)

    with pytest.raises(ValueError, match="base-layer route row totals differ"):
        routes.validate_fragments(
            fragments,
            {
                "request_expert_batch_rows": routes.BASE_LAYERS,
                "request_expert_batch_routes": routes.BASE_LAYERS * routes.TOP_K,
            },
        )


def test_layer_summary_reports_uniform_distribution() -> None:
    summary = routes.layer_summary(3, [6] * routes.EXPERTS, routes.EXPERTS)

    assert summary["routes"] == routes.EXPERTS * routes.TOP_K
    assert summary["zero_hit_experts"] == 0
    assert summary["min"] == summary["max"] == 6
    assert summary["max_to_mean"] == 1.0
    assert summary["normalized_entropy"] == pytest.approx(1.0)
