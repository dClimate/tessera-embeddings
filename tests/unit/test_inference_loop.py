"""Tests for the v1.1 bucketed inference loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tessera_embeddings.inference.data_loading import ChunkData
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.inference import (
    _prepare_batch,
    _prepare_gpu,
    _transfer_and_forward,
    _write_quantized_rows,
    run_inference,
)


class TestRunInference:
    """Tests for the inference loop output shape and zero-fill behavior."""

    def test_output_shape(self, sample_chunk_data, inference_config, test_model):
        """Output is (H, W, save_dim) int8 with (H, W) float32 scales."""
        h, w = 8, 8
        chunk = sample_chunk_data(height=h, width=w, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        result = run_inference(test_model, dataset, inference_config, torch.device("cpu"))

        # The tiny test config uses representation_dim=32, so the inference loop
        # saves min(EMBEDDING_DIM=128, 32) = 32 dims. Validate shape consistency.
        assert result.embeddings.ndim == 3
        assert result.embeddings.shape[:2] == (h, w)
        assert result.embeddings.dtype == np.int8
        assert result.scales.shape == (h, w)
        assert result.scales.dtype == np.float32
        assert result.embeddings_std is None

    def test_invalid_pixels_are_zero(self, inference_config, test_model):
        """Pixels that can't be generated get 0 in every band and a NaN scale.

        All-zero inputs mean no valid pixels — this is the same shape as a pixel
        outside the ROI but inside the bbox, which ingest zeroes across all bands
        so it fails the nonzero validity check here.
        """
        h, w, t = 4, 4, 10
        chunk = ChunkData(
            s2_bands=np.zeros((t, h, w, 10), dtype=np.uint16),
            s2_masks=np.zeros((t, h, w), dtype=np.int32),
            s2_doys=np.arange(1, t + 1, dtype=np.int32),
            s1_asc_bands=np.zeros((t, h, w, 2), dtype=np.uint16),
            s1_asc_doys=np.arange(1, t + 1, dtype=np.int32),
            s1_desc_bands=np.zeros((t, h, w, 2), dtype=np.uint16),
            s1_desc_doys=np.arange(1, t + 1, dtype=np.int32),
            height=h,
            width=w,
            s2_obs_count=np.zeros((h, w), dtype=np.uint16),
        )
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        result = run_inference(test_model, dataset, inference_config, torch.device("cpu"))

        np.testing.assert_array_equal(result.embeddings, 0)
        assert np.isnan(result.scales).all()

    def test_partial_validity_marks_only_invalid_pixels(self, sample_chunk_data, inference_config, test_model):
        """A chunk mixing valid and can't-generate pixels marks exactly the latter.

        Valid pixels get a finite scale and (with random model weights) nonzero
        embeddings; the zeroed-out pixels keep 0 embeddings and a NaN scale.
        """
        h, w = 4, 4
        chunk = sample_chunk_data(height=h, width=w, t_s2=10, t_s1a=5, t_s1d=5)
        # Zero out the bottom-right pixel across every S2 band/time so it fails
        # the nonzero validity check — the in-bbox-outside-ROI case.
        chunk.s2_bands[:, h - 1, w - 1, :] = 0
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        result = run_inference(test_model, dataset, inference_config, torch.device("cpu"))

        invalid = np.isnan(result.scales)
        assert invalid[h - 1, w - 1]
        # Invalid pixels are zero in every band; valid pixels have finite scales.
        np.testing.assert_array_equal(result.embeddings[invalid], 0)
        assert np.isfinite(result.scales[~invalid]).all()

    def test_nonzero_embeddings_for_valid_pixels(self, sample_chunk_data, inference_config, test_model):
        """Valid pixels produce non-zero embeddings (random model weights)."""
        chunk = sample_chunk_data(height=5, width=5, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        result = run_inference(test_model, dataset, inference_config, torch.device("cpu"))

        assert np.any(result.embeddings != 0)

    def test_pipelined_output_matches_serial_reference(self, sample_chunk_data, inference_config, test_model):
        """The pipelined loop is byte-identical to a strictly-serial reference.

        run_inference overlaps each batch's quantize+scatter-write with the next
        batch's forward on a worker thread. Quantization is per-row and each
        sub-batch writes only its own (disjoint) global_idxs, so the reordering
        must not change the result. Build a serial reference by driving the same
        two primitives (_transfer_and_forward → _quantize_and_write) in strict
        order and assert exact equality — this guards the async global_idxs
        handoff. Uses a chunk large enough (with batch_size=16) to span many
        sub-batches across both buckets so the overlap path is actually taken.
        """
        from tessera_embeddings.inference.inference import _transfer_and_forward, _write_quantized_rows

        h, w = 24, 24
        chunk = sample_chunk_data(height=h, width=w, t_s2=10, t_s1a=5, t_s1d=5)
        device = torch.device("cpu")
        save_dim = inference_config.representation_dim

        # Serial reference: same primitives, no pipelining.
        dataset_ref = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        ref_q = np.zeros((h * w, save_dim), dtype=np.int8)
        ref_scales = np.full(h * w, np.nan, dtype=np.float32)
        with torch.no_grad():
            for bucket_key, pixel_indices in dataset_ref.iter_buckets(largest_first=True):
                n = int(pixel_indices.size)
                for start in range(0, n, inference_config.batch_size):
                    end = min(start + inference_config.batch_size, n)
                    batch, _ = _prepare_batch(dataset_ref, bucket_key, start, end)
                    q_host, scales_host, global_idxs, _, _ = _transfer_and_forward(
                        test_model, batch, bucket_key, device, None, save_dim
                    )
                    _write_quantized_rows(q_host, scales_host, global_idxs, ref_q, ref_scales)
        ref_emb = ref_q.reshape(h, w, save_dim)
        ref_scales_2d = ref_scales.reshape(h, w)

        # Ensure the reference actually spanned multiple sub-batches (else the
        # test wouldn't exercise the overlap it claims to).
        total_valid = sum(int(pi.size) for _, pi in dataset_ref.iter_buckets())
        assert total_valid > inference_config.batch_size, "chunk too small to exercise pipelining"

        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        result = run_inference(test_model, dataset, inference_config, device)

        np.testing.assert_array_equal(result.embeddings, ref_emb)
        # NaN-aware comparison for scales (invalid pixels are NaN in both).
        np.testing.assert_array_equal(np.isnan(result.scales), np.isnan(ref_scales_2d))
        finite = ~np.isnan(ref_scales_2d)
        np.testing.assert_array_equal(result.scales[finite], ref_scales_2d[finite])

    def test_compute_std_is_coerced_off(self, sample_chunk_data, inference_config, test_model):
        """compute_std is silently forced off under v1.1; embeddings_std is None."""
        h, w = 8, 8
        # Try to turn it on; the dataclass __post_init__ forces it back to False.
        inference_config.compute_std = True
        chunk = sample_chunk_data(height=h, width=w, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        result = run_inference(test_model, dataset, inference_config, torch.device("cpu"))
        assert result.embeddings_std is None


class TestPrepareGpu:
    """Tests for _prepare_gpu helper."""

    def test_cpu_returns_none(self, test_model):
        """On CPU, returns None (inputs stay float32) and leaves the model float32."""
        reduced_dtype = _prepare_gpu(test_model, torch.device("cpu"))

        assert reduced_dtype is None
        param = next(test_model.parameters())
        assert param.dtype == torch.float32

    def test_cpu_no_side_effects(self, test_model):
        """Calling _prepare_gpu on CPU is safe (no CUDA required)."""
        params_before = {n: p.clone() for n, p in test_model.named_parameters()}

        _prepare_gpu(test_model, torch.device("cpu"))

        for name, param in test_model.named_parameters():
            torch.testing.assert_close(param, params_before[name])

    def test_dtype_conversion_is_idempotent(self, test_model, monkeypatch):
        """Re-running on an already-converted model skips the re-cast.

        Actors reuse one persistent model across every strip, so _prepare_gpu
        is invoked once per strip. The cast must happen on the first call and
        be skipped thereafter — hoisting the one-time cost off the per-strip
        path. Simulated with a CUDA device + bf16 support so no GPU is needed.
        """
        device = torch.device("cuda")
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        # Profiling/diagnostics helpers touch real CUDA; stub them out.
        monkeypatch.setattr("tessera_embeddings.inference.inference.log_cuda_diagnostics", lambda *a, **k: None)
        monkeypatch.setattr("tessera_embeddings.inference.inference.log_autocast_dtype_probe", lambda *a, **k: None)

        casts = {"count": 0}
        real_bfloat16 = test_model.bfloat16

        def counting_bfloat16():
            casts["count"] += 1
            return real_bfloat16()

        monkeypatch.setattr(test_model, "bfloat16", counting_bfloat16)

        first = _prepare_gpu(test_model, device)
        second = _prepare_gpu(test_model, device)

        assert first == torch.bfloat16
        assert second == torch.bfloat16
        assert casts["count"] == 1  # converted once, skipped on the second call


class TestProcessSubBatch:
    """Tests for the split _transfer_and_forward / _write_quantized_rows primitives."""

    def test_transfer_and_forward_returns_quantized_host_rows(self, sample_chunk_data, inference_config, test_model):
        """_transfer_and_forward returns on-device-quantized int8 rows + scales on the host."""
        chunk = sample_chunk_data(height=5, width=5, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        device = torch.device("cpu")
        save_dim = inference_config.representation_dim

        bucket_key, pixel_indices = next(iter(dataset.iter_buckets()))
        end = min(inference_config.batch_size, int(pixel_indices.size))
        batch, _ = _prepare_batch(dataset, bucket_key, 0, end)

        with torch.no_grad():
            q_host, scales_host, global_idxs, transfer_secs, forward_secs = _transfer_and_forward(
                test_model, batch, bucket_key, device, None, save_dim, config=inference_config
            )

        assert q_host.shape == (end, save_dim)
        assert q_host.dtype == np.int8
        assert scales_host.shape == (end,)
        assert scales_host.dtype == np.float32
        assert global_idxs.shape == (end,)
        assert transfer_secs >= 0
        assert forward_secs >= 0

    def test_write_quantized_rows_writes_to_flat_out(self, sample_chunk_data, inference_config, test_model):
        """_write_quantized_rows scatters non-zero int8 rows into the output buffers."""
        chunk = sample_chunk_data(height=5, width=5, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        device = torch.device("cpu")
        h, w = dataset.H, dataset.W
        save_dim = inference_config.representation_dim
        flat_q = np.zeros((h * w, save_dim), dtype=np.int8)
        flat_scales = np.full(h * w, np.nan, dtype=np.float32)

        bucket_key, pixel_indices = next(iter(dataset.iter_buckets()))
        end = min(inference_config.batch_size, int(pixel_indices.size))
        batch, _ = _prepare_batch(dataset, bucket_key, 0, end)

        with torch.no_grad():
            q_host, scales_host, global_idxs, _, _ = _transfer_and_forward(
                test_model, batch, bucket_key, device, None, save_dim, config=inference_config
            )
        _write_quantized_rows(q_host, scales_host, global_idxs, flat_q, flat_scales)

        assert np.any(flat_q != 0)
        # Only this bucket's pixels were written; their scales are now finite.
        assert np.isfinite(flat_scales[global_idxs]).all()


class TestThroughputAccounting:
    """Tests for the tokens/s + effective-TFLOPS accounting in run_inference."""

    def test_final_log_reports_tokens_and_tflops(self, sample_chunk_data, inference_config, test_model, caplog):
        """The completion log carries density-neutral throughput units.

        px/s conflates sequence length across chunks, so the summary must also
        report tok/sec and effective TFLOPS for cross-chunk comparison.
        """
        import logging

        chunk = sample_chunk_data(height=8, width=8, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)

        with caplog.at_level(logging.INFO, logger="tessera_embeddings.inference.inference"):
            run_inference(test_model, dataset, inference_config, torch.device("cpu"))

        summary = next(r.message for r in caplog.records if r.message.startswith("Inference complete"))
        assert "tok/sec avg" in summary
        assert "eff TFLOPS avg" in summary


class TestPipelinedGpuLoop:
    """The async two-deep CUDA loop must match the synchronous serial loop.

    Both loops run the same ops in the same order on the same stream, so results
    are bit-identical; the pipeline only changes when the host waits. This is the
    only coverage of the pinned-buffer / CUDA-event / two-deep-drain / backbone-
    stream-ordering path — the CPU tests exercise `_serial_loop` only. Skips when
    no GPU is present (CI), runs on GPU boxes.

    **THIS PATH HAS NO CI COVERAGE, BY DECISION (2026-08-25).** No GPU runner is available
    to the project and one will not be provisioned, so the skip below fires on every CI run
    and this test is verified BY HAND. It keeps its ``skipif`` rather than taking
    ``@pytest.mark.gpu`` precisely so it still reports as a skip: the marker is filtered out
    by the default ``addopts``, and the test would then vanish from the run entirely, leaving
    nothing to notice the gap by.

    Run it on any CUDA machine after touching the pipelined loop, the actor pool or the
    chunk-staging path, and once before a campaign starts::

        uv sync --all-extras --frozen
        uv run pytest tests/unit/test_inference_loop.py -k pipelined -v

    See ``tests/README.md`` → Roadmap 2.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the pipelined GPU loop")
    def test_pipelined_matches_serial(self, sample_chunk_data, inference_config, test_model, monkeypatch):
        import os

        from tessera_embeddings.inference.inference import _SERIAL_LOOP_ENV

        chunk = sample_chunk_data(height=16, width=16, t_s2=12, t_s1a=6, t_s1d=6)
        device = torch.device("cuda")
        model = test_model.to(device)

        # Serial reference (escape-hatch env forces _serial_loop even on CUDA).
        monkeypatch.setenv(_SERIAL_LOOP_ENV, "1")
        ds_serial = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        serial = run_inference(model, ds_serial, inference_config, device)

        # Default path → _pipelined_gpu_loop.
        monkeypatch.delenv(_SERIAL_LOOP_ENV, raising=False)
        assert not os.environ.get(_SERIAL_LOOP_ENV)
        ds_pipe = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        pipelined = run_inference(model, ds_pipe, inference_config, device)

        # Same ops, same order, same stream → bit-identical int8 and scales.
        np.testing.assert_array_equal(pipelined.embeddings, serial.embeddings)
        np.testing.assert_array_equal(pipelined.scales, serial.scales)


class TestCardCeilings:
    """The per-card ceiling behind the EFFECTIVE TFLOPS line's verdict.

    The line used to hardcode the L40S's numbers and grade against absolute
    TFLOPS bands, which mislabels any other card: an A10G doing respectable work
    lands near 26 TFLOPS and the old `<12 poor / >20 active` bands would have
    called that saturated while calling a starved L40S the same.
    """

    def test_the_longest_matching_card_name_wins(self) -> None:
        """`"NVIDIA L40S"` contains `"L4"`, and resolving it to the L4 entry would
        understate the card's ceiling by 3x and its bandwidth by 2.9x."""
        from tessera_embeddings.inference.profiling import _card_ceiling

        assert _card_ceiling("NVIDIA L40S")[0] == "L40S"
        assert _card_ceiling("NVIDIA L4")[0] == "L4"
        assert _card_ceiling("NVIDIA A10G")[0] == "A10G"

    def test_an_unknown_card_returns_none_rather_than_a_default(self) -> None:
        """A wrong ceiling turns a saturated GPU into "poor utilization", or the reverse."""
        from tessera_embeddings.inference.profiling import _card_ceiling

        assert _card_ceiling("NVIDIA H100 80GB HBM3") is None

    def test_the_verified_bandwidth_figures(self) -> None:
        """AWS publishes no GPU memory bandwidth at all, so these came from the
        vendors' datasheets and are the load-bearing half of the table."""
        from tessera_embeddings.inference.profiling import _CARD_CEILINGS

        assert _CARD_CEILINGS["L40S"][1] == 864.0
        assert _CARD_CEILINGS["A10G"][1] == 600.0
        assert _CARD_CEILINGS["L4"][1] == 300.0

    def test_the_ceiling_is_the_figure_that_cannot_be_exceeded(self) -> None:
        """A measured A10G reached 35.9 TFLOPS against a halved (FP32-accumulate)
        figure of 35.0, which cannot happen — so the table carries the vendor's
        quoted dense figure instead, and the fraction is an index, not a
        utilisation. Pinned so a future 'correction' back to the halved values
        has to argue with this."""
        from tessera_embeddings.inference.profiling import _CARD_CEILINGS

        assert _CARD_CEILINGS["A10G"][0] == 70.0
        assert _CARD_CEILINGS["L4"][0] == 121.0
        assert _CARD_CEILINGS["L40S"][0] == 362.0
