"""Unit tests for the inference worker's background resource monitor.

Focused on ``_get_gpu_stats``, which was wrong on any host with more than one
GPU and could not have been caught until one was tried: ``nvidia-smi`` does not
honour ``CUDA_VISIBLE_DEVICES``, so without ``-i`` it answers for every GPU on
the box. These tests fake the subprocess, so they need no GPU.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from tessera_embeddings.inference import resource_monitor as rm

#: One nvidia-smi CSV row in the field order the query asks for:
#: utilization.gpu, utilization.memory, memory.used, memory.total, temperature.gpu, power.draw
_ROW = "97, 62, 28451, 45776, 71, 331.42"


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout=stdout, stderr="")


class TestGetGpuStats:
    """One GPU's stats, attributed to the actor that owns that GPU."""

    def test_passes_the_actors_own_index_to_nvidia_smi(self) -> None:
        """``-i <index>`` is the fix. Without it a 4-GPU host answers four times over."""
        seen: dict = {}

        def _run(cmd, **kwargs):
            seen["cmd"] = cmd
            return _completed(_ROW + "\n")

        with patch.object(rm.subprocess, "run", _run):
            stats = rm._get_gpu_stats("2")
        assert seen["cmd"][:3] == ["nvidia-smi", "-i", "2"]
        assert stats == {
            "gpu_util": "97%",
            "mem_util": "62%",
            "mem_used": "28451 MiB",
            "mem_total": "45776 MiB",
            "temp": "71C",
            "power": "331.42W",
        }

    def test_omits_the_flag_when_no_index_is_known(self) -> None:
        """Correct only on a one-GPU host, which is why the multi-row guard below exists."""
        seen: dict = {}

        def _run(cmd, **kwargs):
            seen["cmd"] = cmd
            return _completed(_ROW + "\n")

        with patch.object(rm.subprocess, "run", _run):
            assert rm._get_gpu_stats(None) is not None
        assert "-i" not in seen["cmd"]

    def test_refuses_multi_row_output_instead_of_misattributing_it(self) -> None:
        """The original defect, pinned.

        The old code split the whole multi-line output on commas, so four rows filled
        six field names from the first row plus a newline-joined boundary: every actor
        on the host reported GPU 0, with fields sliding out of alignment. Several rows
        now mean the ``-i`` filter did not apply, and the honest answer to "which GPU
        is this?" is no answer — a monitor line that silently omits GPU stats is
        recoverable; one that attributes another GPU's numbers to this actor is not.
        """
        four_rows = "\n".join([_ROW] * 4)
        with patch.object(rm.subprocess, "run", lambda *a, **k: _completed(four_rows)):
            assert rm._get_gpu_stats("0") is None

    def test_refuses_a_row_with_the_wrong_field_count(self) -> None:
        with patch.object(rm.subprocess, "run", lambda *a, **k: _completed("97, 62, 28451")):
            assert rm._get_gpu_stats("0") is None

    @pytest.mark.parametrize(
        "outcome",
        [
            _completed("", returncode=9),
            FileNotFoundError("nvidia-smi"),
            subprocess.TimeoutExpired("nvidia-smi", 5),
        ],
    )
    def test_absent_or_failing_nvidia_smi_yields_none(self, outcome) -> None:
        """A CPU host, a driver hiccup, a wedged call — none of these may raise into the actor."""

        def _run(*a, **k):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        with patch.object(rm.subprocess, "run", _run):
            assert rm._get_gpu_stats("0") is None

    def test_field_names_and_query_order_agree(self) -> None:
        """The parse maps by position in one list, so the list and the query must match.

        A misaligned parse produces plausible-looking numbers under the wrong names,
        which is the failure mode that survived undetected on multi-GPU hosts.
        """
        seen: dict = {}

        def _run(cmd, **kwargs):
            seen["query"] = next(a for a in cmd if a.startswith("--query-gpu="))
            return _completed(_ROW)

        with patch.object(rm.subprocess, "run", _run):
            rm._get_gpu_stats("0")
        queried = seen["query"].removeprefix("--query-gpu=").split(",")
        assert len(queried) == len(rm._GPU_FIELDS)
        assert queried == [
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
        ]


class TestResourceMonitorLine:
    """The RESOURCES line names which GPU it describes."""

    def test_stamps_the_gpu_index_so_four_actors_are_distinguishable(self) -> None:
        """Four actors on one host otherwise emit four identical RESOURCES lines."""
        monitor = rm.ResourceMonitor(interval_sec=1, gpu_index="3")
        logged: list[str] = []
        with (
            patch.object(rm, "_get_cpu_mem_stats", return_value={"load_avg": "1 2 3"}),
            patch.object(
                rm,
                "_get_gpu_stats",
                return_value={
                    "gpu_util": "97%",
                    "mem_used": "1 MiB",
                    "mem_total": "2 MiB",
                    "temp": "70C",
                    "power": "300W",
                },
            ) as gpu,
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        gpu.assert_called_once_with("3")
        assert "gpu_idx=3" in logged[0]

    def test_no_index_means_no_stamp(self) -> None:
        """A one-GPU host gains nothing from the field, so it is not added."""
        monitor = rm.ResourceMonitor(interval_sec=1)
        logged: list[str] = []
        with (
            patch.object(rm, "_get_cpu_mem_stats", return_value={"load_avg": "1 2 3"}),
            patch.object(
                rm,
                "_get_gpu_stats",
                return_value={
                    "gpu_util": "97%",
                    "mem_used": "1 MiB",
                    "mem_total": "2 MiB",
                    "temp": "70C",
                    "power": "300W",
                },
            ),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        assert "gpu_idx" not in logged[0]
