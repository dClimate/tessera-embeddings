"""The observe_cluster phase parser reports the authoritative chunk-total valid_px.

Regression for the 2026-07-17 mis-measurement: a multi-strip chunk's per-strip
``ds`` lines under-count valid pixels (a no-valid-pixel strip emits no ds line),
so the parser must take valid_px + px/s from the actor's chunk-total "complete:
N valid pixels" line, not the ds sum.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inference_perf" / "observe_cluster.py"

# A 2-strip chunk whose per-strip ds sum (100) disagrees with the actor's
# chunk-total complete line (2500) — the exact multi-strip under-count. Built
# from a line list (implicit concatenation on the long one) so no source line
# exceeds the length limit while the emitted log lines stay single-line.
_FAKE_LOG = "\n".join(
    [
        "2026-07-17 10:00:00,000 host INFO Chunk chunk_9_9: T_kept=63 -> strip_h=1000 -> 2 strip(s) [prefetch=True]",
        "2026-07-17 10:00:01,000 host INFO MosaicChunkInferenceDataset: 100 valid pixels out of 200 total (50.0%) in 4 buckets",  # noqa: E501
        "2026-07-17 10:00:02,000 host INFO Starting v1.1 inference: 4 buckets",
        (
            "2026-07-17 10:00:10,000 host INFO Inference complete: 100 valid pixels, "
            "output shape (1000, 2000, 128), dtype int8, 8.0s total, 12 px/sec avg"
        ),
        "2026-07-17 10:00:12,000 host INFO Chunk chunk_9_9 complete: 2500 valid pixels, 12.0s",
    ]
)


def _load_observe_cluster():
    spec = importlib.util.spec_from_file_location("observe_cluster", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase_parser_valid_px_from_cdone(tmp_path: Path) -> None:
    mod = _load_observe_cluster()
    logdir = tmp_path / "logs"
    logdir.mkdir()
    (logdir / "worker-abc.err").write_text(_FAKE_LOG)

    parser = mod.PHASE_PARSER.replace("/tmp/ray/session_latest/logs/worker-*.err", str(logdir / "worker-*.err"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(parser, "<phase_parser>", "exec"), {})
    lines = buf.getvalue().splitlines()

    header, row = lines[0].split("\t"), lines[1].split("\t")
    cols = dict(zip(header, row, strict=True))
    # valid_px is the authoritative chunk total (2500), NOT the ds sum (100).
    assert cols["valid_px"] == "2500"
    # px/s is end-to-end: 2500 / 12.0s ~= 208.
    assert cols["px/s"] == "208"
    assert cols["strips"] == "2"
