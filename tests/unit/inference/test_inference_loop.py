"""Tests for the v1.1 bucketed inference loop."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tessera_embeddings.inference.data_loading import ChunkData
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.inference import (
    _SERIAL_LOOP_ENV,
    _prepare_batch,
    _prepare_gpu,
    _transfer_and_forward,
    _write_quantized_rows,
    run_inference,
)


class TestRunInference:
    """The inference loop's zero-fill behaviour, and the pipelined loop against a serial reference."""

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
        assert np.any(result.embeddings[~invalid] != 0)

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
                        test_model, batch, bucket_key, device, save_dim
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


class TestPrepareGpu:
    """Tests for _prepare_gpu helper."""

    def test_cpu_leaves_the_model_in_float32(self, test_model):
        """On CPU there is no reduced-precision cast at all."""
        _prepare_gpu(test_model, torch.device("cpu"))

        param = next(test_model.parameters())
        assert param.dtype == torch.float32

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

        _prepare_gpu(test_model, device)
        _prepare_gpu(test_model, device)

        assert next(test_model.parameters()).dtype == torch.bfloat16
        assert casts["count"] == 1  # converted once, skipped on the second call


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
    chunk-staging path, and once before a campaign starts. **The setup is the part that goes
    wrong**: ``uv sync --frozen`` installs the locked CPU wheel, and ``uv run`` re-syncs from
    the lockfile by default, so a bare invocation SKIPS rather than verifying anything::

        uv sync --all-extras --frozen
        uv pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
        uv run --no-sync python -c "import torch; assert torch.cuda.is_available()"
        uv run --no-sync pytest tests/unit/inference/test_inference_loop.py -k pipelined -v

    **Pin the version.** An unconstrained ``uv pip install torch`` is already satisfied by the
    locked CPU wheel and does nothing — verified, uv reports "Would make no changes" — leaving you
    on ``+cpu`` with a passing-looking setup and a skipped test. ``pytest`` exits 0 on a skip, so
    only a PASS verifies anything here.

    See ``tests/README.md`` → Roadmap 2 for the rest of the setup, and check
    ``https://download.pytorch.org/whl/<cuXXX>/torch/`` before changing that pin: the build must
    exist and must match your driver.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the pipelined GPU loop")
    def test_pipelined_matches_serial(self, sample_chunk_data, inference_config, test_model, monkeypatch):
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
