"""Tests for the v1.1 bucketed inference loop."""

from __future__ import annotations

import numpy as np
import torch

from tessera_embeddings.inference.data_loading import ChunkData
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.inference import (
    _BatchTimings,
    _prepare_batch,
    _prepare_gpu,
    _run_gpu_sub_batch,
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


class TestProcessSubBatch:
    """Tests for the split _prepare_batch / _run_gpu_sub_batch helpers."""

    def test_returns_batch_timings(self, sample_chunk_data, inference_config, test_model):
        """_run_gpu_sub_batch returns _BatchTimings with all non-negative values."""
        chunk = sample_chunk_data(height=5, width=5, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        device = torch.device("cpu")
        h, w = dataset.H, dataset.W
        save_dim = inference_config.representation_dim
        flat_q = np.zeros((h * w, save_dim), dtype=np.int8)
        flat_scales = np.full(h * w, np.nan, dtype=np.float32)

        bucket_key, pixel_indices = next(iter(dataset.iter_buckets()))
        end = min(inference_config.batch_size, int(pixel_indices.size))
        batch, get_batch_secs = _prepare_batch(dataset, bucket_key, 0, end)

        with torch.no_grad():
            bt = _run_gpu_sub_batch(
                test_model,
                batch,
                bucket_key,
                device,
                None,
                flat_q,
                flat_scales,
                get_batch_secs,
                config=inference_config,
                save_dim=save_dim,
            )

        assert isinstance(bt, _BatchTimings)
        assert bt.get_batch >= 0
        assert bt.transfer >= 0
        assert bt.forward >= 0
        assert bt.postprocess >= 0

    def test_writes_to_flat_out(self, sample_chunk_data, inference_config, test_model):
        """_run_gpu_sub_batch writes non-zero values into the output buffers."""
        chunk = sample_chunk_data(height=5, width=5, t_s2=10, t_s1a=5, t_s1d=5)
        dataset = MosaicChunkInferenceDataset(chunk, num_obs_checkpoints=inference_config.num_obs_checkpoints)
        device = torch.device("cpu")
        h, w = dataset.H, dataset.W
        save_dim = inference_config.representation_dim
        flat_q = np.zeros((h * w, save_dim), dtype=np.int8)
        flat_scales = np.full(h * w, np.nan, dtype=np.float32)

        bucket_key, pixel_indices = next(iter(dataset.iter_buckets()))
        end = min(inference_config.batch_size, int(pixel_indices.size))
        batch, get_batch_secs = _prepare_batch(dataset, bucket_key, 0, end)

        with torch.no_grad():
            _run_gpu_sub_batch(
                test_model,
                batch,
                bucket_key,
                device,
                None,
                flat_q,
                flat_scales,
                get_batch_secs,
                config=inference_config,
                save_dim=save_dim,
            )

        assert np.any(flat_q != 0)
