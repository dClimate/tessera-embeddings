"""The S2 flow pipelines dates on its own, up to ``MAX_PIPELINE_DATES_WORKERS``.

Nobody sets pipelining: the flow overlaps each date's preparation with the previous date's write
on every fleet up to the bound, and above it prepares dates one at a time, saying so once. Widths
derive from the constant so a re-calibrated bound cannot leave a test asserting a stale one.
"""

from __future__ import annotations

import logging

import pytest

from tessera_embeddings.orchestration.prefect.flows.ingest_s2_roi_reflectance import (
    MAX_PIPELINE_DATES_WORKERS,
    _pipelines_dates,
)

BOUND = MAX_PIPELINE_DATES_WORKERS


@pytest.mark.parametrize(
    ("max_workers", "use_local", "expected"),
    [(60, False, True), (BOUND, False, True), (BOUND + 1, False, False), (BOUND + 100, True, True)],
    ids=["narrow", "at-the-inclusive-bound", "above-the-bound", "local-cluster-ignores-width"],
)
def test_pipelines_unless_the_fargate_fleet_is_wider_than_the_bound(
    max_workers: int, use_local: bool, expected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    log = logging.getLogger("s2-pipeline-width")
    with caplog.at_level(logging.INFO, logger=log.name):
        assert _pipelines_dates(use_local=use_local, max_workers=max_workers, log=log) is expected
    # Declining is the only case worth a line, and that line names the width behind it.
    assert [str(max_workers) in r.getMessage() for r in caplog.records] == ([] if expected else [True])
