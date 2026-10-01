"""The span drain's slicing: a failed slice keeps the progress before it, and nothing is lost or doubled."""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from tessera_embeddings.providers import frisky as frisky_engine

S = 1_000_000_000


def _spans(n: int) -> list[dict[str, Any]]:
    """One worker.exec.call span ending every second from t=1 s."""
    return [
        {"name": "worker.exec.call", "worker": "w", "span_id": i, "start_ns": i * S - 1, "end_ns": i * S}
        for i in range(1, n + 1)
    ]


def test_a_failed_slice_keeps_earlier_slices_and_the_next_drain_fills_the_gap(tmp_path, monkeypatch) -> None:
    buffered = _spans(60)
    calls = {"n": 0, "fail_at": -1}  # armed below, after the first drain

    def query_spans(*, name: str, start_ns: int | None, end_ns: int | None, **_: Any) -> list[dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] == calls["fail_at"]:
            raise OSError("HTTP Error 504: Gateway Timeout")
        if name != "worker.exec":
            return []
        return [
            s
            for s in buffered
            if (start_ns is None or s["end_ns"] >= start_ns) and (end_ns is None or s["start_ns"] <= end_ns)
        ]

    monkeypatch.setattr(frisky_engine.frisky, "query_spans", query_spans)
    monkeypatch.setattr(frisky_engine, "_SPAN_DRAIN_SLICE_NS", 10 * S)
    monkeypatch.setattr(frisky_engine, "_SPAN_DRAIN_LAG_NS", 0)
    clock = iter([20 * S, 61 * S, 61 * S])
    monkeypatch.setattr(frisky_engine.time, "time_ns", lambda: next(clock))
    drain = frisky_engine._SpanDrain("http://unused", str(tmp_path))

    drain.drain()  # the first drain takes everything before t=20 s in one request
    n_names = len(frisky_engine.SPAN_DRAIN_NAMES)
    calls["fail_at"] = calls["n"] + n_names + 1  # fail the second slice of the next drain
    with pytest.raises(OSError):
        drain.drain()
    assert drain.since_ns == 30 * S, "the slice before the failure should count"
    drain.drain()  # retries from 30 s to now

    parts = sorted(tmp_path.glob("spans/part-*.json.gz"))
    got = [s["span_id"] for p in parts for s in json.loads(gzip.decompress(p.read_bytes()))]
    assert sorted(got) == [s["span_id"] for s in buffered if s["end_ns"] < 61 * S]
    assert len(got) == len(set(got)), "a span was written twice"
