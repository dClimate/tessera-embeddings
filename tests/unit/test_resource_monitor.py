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
                    "mem_util": "44%",
                    "mem_used": "1 MiB",
                    "mem_total": "2 MiB",
                    "temp": "70C",
                    "power": "300W",
                },
            ) as gpu,
            patch.object(rm, "_get_gpu_extra_stats", return_value={}),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        gpu.assert_called_once_with("3")
        assert "gpu_idx=3" in logged[-1]

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
                    "mem_util": "44%",
                    "mem_used": "1 MiB",
                    "mem_total": "2 MiB",
                    "temp": "70C",
                    "power": "300W",
                },
            ),
            patch.object(rm, "_get_gpu_extra_stats", return_value={}),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        assert "gpu_idx" not in logged[-1]


class TestGpuExtraStats:
    """Clock, PCIe link and throttle state — each one optional, none load-bearing.

    The whole point of the split from :func:`_get_gpu_stats` is that a driver
    which rejects one of these field names must not cost us utilisation, VRAM,
    temperature and power as collateral. nvidia-smi fails a whole query on one
    unknown field name, so that collateral is the default behaviour unless the
    calls are separated.
    """

    def setup_method(self) -> None:
        rm._throttle_field = None

    def teardown_method(self) -> None:
        rm._throttle_field = None

    def test_reports_clock_pcie_and_throttle_when_all_are_supported(self) -> None:
        with patch.object(rm, "_query_gpu", side_effect=[["2100", "4", "16"], ["0x0000"], ["0x0000"]]):
            stats = rm._get_gpu_extra_stats("0")
        assert stats == {"sm_clock": "2100MHz", "pcie": "gen4x16", "throttle": "0x0000"}

    def test_falls_back_to_the_older_throttle_field_name(self) -> None:
        """nvidia-smi renamed `clocks_throttle_reasons.*` to `clocks_event_reasons.*`."""
        calls: list[tuple[str, ...]] = []

        def fake(_index, fields):
            calls.append(fields)
            if fields == ("clocks_event_reasons.active",):
                return None
            if fields == ("clocks_throttle_reasons.active",):
                return ["0x0004"]
            return ["1800", "4", "8"]

        with patch.object(rm, "_query_gpu", side_effect=fake):
            stats = rm._get_gpu_extra_stats(None)
        assert stats["throttle"] == "0x0004"
        assert ("clocks_event_reasons.active",) in calls
        assert rm._throttle_field == "clocks_throttle_reasons.active"

    def test_no_throttle_field_at_all_leaves_the_other_metrics_intact(self) -> None:
        def fake(_index, fields):
            return None if "reasons" in fields[0] else ["1800", "4", "8"]

        with patch.object(rm, "_query_gpu", side_effect=fake):
            stats = rm._get_gpu_extra_stats(None)
        assert "throttle" not in stats
        assert stats["sm_clock"] == "1800MHz"
        assert rm._throttle_field == ""

    def test_a_failing_extra_query_returns_an_empty_dict_not_an_error(self) -> None:
        with patch.object(rm, "_query_gpu", return_value=None):
            assert rm._get_gpu_extra_stats("0") == {}


class TestHostRamPeak:
    """A high-water mark, because the emitted instantaneous figure is not one.

    The per-actor RAM budget is sized to leave ~0.9 GB under a 60% ceiling for
    spikes SHORTER than the 30-second emit cadence — so the number the line has
    always carried is systematically below the peak it is meant to police.
    """

    def test_the_peak_survives_a_later_lower_sample(self) -> None:
        monitor = rm.ResourceMonitor(interval_sec=30)
        with patch.object(rm, "read_host_ram_gib", side_effect=[(10.0, 32.0), (18.0, 32.0), (11.0, 32.0)]):
            for _ in range(3):
                monitor._sample_ram()
        assert monitor.peak_host_ram_gib() == (18.0, 32.0)

    def test_nothing_sampled_reads_as_none_not_zero(self) -> None:
        """Zero would say "measured, and it was nothing" — the wrong claim off Linux."""
        monitor = rm.ResourceMonitor(interval_sec=30)
        assert monitor.peak_host_ram_gib() is None
        with patch.object(rm, "read_host_ram_gib", return_value=None):
            monitor._sample_ram()
        assert monitor.peak_host_ram_gib() is None

    def test_reset_reports_not_sampled_rather_than_a_measured_zero(self) -> None:
        """REPLACES ``test_reset_drops_the_mark_but_keeps_the_total``, which asserted the
        opposite — that a reset leaves ``(0.0, total)`` readable.

        That was the defect written down as the contract. ``peak_host_ram_gib`` signals "not
        sampled" with a zero TOTAL and its docstring says a caller "must not read that as
        zero", but a reset produced exactly the state the docstring forbids: peak cleared,
        total retained. ``_host_fields`` then published a measured-looking
        ``host_ram_peak_gib: 0.0`` for any chunk that finished inside one 2 s sampling
        interval or was skipped outright, writing false zeros into the per-chunk RAM record.
        Raised in review of PR #150 by four independent readers.
        """
        monitor = rm.ResourceMonitor(interval_sec=30)
        with patch.object(rm, "read_host_ram_gib", return_value=(18.0, 32.0)):
            monitor._sample_ram()
        assert monitor.peak_host_ram_gib() == (18.0, 32.0)
        monitor.reset_peak_host_ram()
        assert monitor.peak_host_ram_gib() is None

    def test_the_sampler_cannot_be_slower_than_the_emit_interval(self) -> None:
        """A default sample_sec above a short interval would emit a stale peak."""
        assert rm.ResourceMonitor(interval_sec=1)._sample_sec == 1
        assert rm.ResourceMonitor(interval_sec=30)._sample_sec == 2.0

    def test_the_peak_is_on_the_line(self) -> None:
        monitor = rm.ResourceMonitor(interval_sec=1)
        logged: list[str] = []
        with patch.object(rm, "read_host_ram_gib", return_value=(19.2, 32.0)):
            monitor._sample_ram()
        with (
            patch.object(rm, "_get_cpu_mem_stats", return_value={"load_avg": "1 2 3"}),
            patch.object(rm, "_get_gpu_stats", return_value=None),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        assert "RAMpeak=19.2/32.0 GB (60%)" in logged[0]


class TestAttributionAndResetSurviveFailure:
    """Two findings from review of PR #150, both about a value that LOOKS measured but is not."""

    def test_the_index_appears_exactly_once_when_the_gpu_sample_succeeds(self) -> None:
        """`RESOURCES` is a key=value line that monitoring parses, so a repeated key is
        a defect: strict parsers break on it and GPU attribution reads as ambiguous.

        The field is appended OUTSIDE the `if gpu` block so a failed nvidia-smi still
        carries attribution. A second append inside that block — which is what a merge
        left here — duplicates it on every successful sample, i.e. almost always.
        """
        monitor = rm.ResourceMonitor(interval_sec=1, gpu_index="2")
        logged: list[str] = []
        stats = {
            "gpu_util": "50%",
            "mem_util": "10%",
            "mem_used": "1 MiB",
            "mem_total": "2 MiB",
            "temp": "40",
            "power": "70 W",
        }
        with (
            patch.object(rm, "_get_cpu_mem_stats", return_value={"load_avg": "1 2 3"}),
            patch.object(rm, "_get_gpu_stats", return_value=stats),
            patch.object(rm, "_get_gpu_extra_stats", return_value={}),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        assert logged[-1].count("gpu_idx=") == 1, logged[-1]

    def test_the_gpu_index_survives_a_failed_gpu_sample(self) -> None:
        """Attribution matters MOST when sampling fails: four actors on one packed host would
        otherwise emit indistinguishable CPU/RAM lines for exactly that interval.
        """
        monitor = rm.ResourceMonitor(interval_sec=1, gpu_index="3")
        logged: list[str] = []
        with (
            patch.object(rm, "_get_cpu_mem_stats", return_value={"load_avg": "1 2 3"}),
            patch.object(rm, "_get_gpu_stats", return_value=None),
            patch.object(rm, "_get_gpu_extra_stats", return_value={}),
            patch.object(rm.logger, "info", lambda fmt, *a: logged.append(fmt % a)),
        ):
            monitor._emit_once()
        assert "gpu_idx=3" in logged[-1], logged[-1]
        assert "GPU=" not in logged[-1], "no GPU stats were available, so none may be claimed"
