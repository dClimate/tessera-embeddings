"""Golden reference test: our v2 Large port == upstream's, on the real checkpoint.

``inference/models/student_v2.py`` is a port; the only thing that makes it
trustworthy is running the *upstream* implementation side by side on the same
weights and the same inputs. ``tests/fixtures/upstream/v2_student_reference.py``
is a verbatim copy of upstream's ``model.py``; this test loads the real
checkpoint into both and asserts the outputs match at fp32.

The 175 MB checkpoint is not in git, so the whole module skips unless
``TESSERA_V2_CKPT`` points at it (download from
``geotessera/TESSERA-V-2.0-2B-L``, file ``ckpt/student_large.pt``)::

    TESSERA_V2_CKPT=/path/to/student_large.pt uv run pytest tests/unit/test_student_v2_golden.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from tessera_embeddings.config.inference import InferenceConfig, band_stats
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.models.builder import build_inference_model

_REFERENCE_PATH = Path(__file__).parents[1] / "fixtures" / "upstream" / "v2_student_reference.py"

# Observed on the real checkpoint (torch 2.12, CPU): bit-identical — max abs diff
# exactly 0.0 at every bucket shape below, since both graphs run the same ops in
# the same order on the same weights. The tolerance exists only to absorb
# cross-platform kernel/threading differences, not structural divergence.
_ATOL = 1e-5


def _checkpoint_path() -> Path | None:
    """Local v2 Large checkpoint from ``TESSERA_V2_CKPT``, if it exists."""
    raw = os.getenv("TESSERA_V2_CKPT")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


pytestmark = pytest.mark.skipif(
    _checkpoint_path() is None,
    reason="set TESSERA_V2_CKPT to the v2 Large checkpoint (geotessera/TESSERA-V-2.0-2B-L) to run",
)


def _load_reference_module():
    """Import the vendored upstream module by path (tests/fixtures is not a package)."""
    spec = importlib.util.spec_from_file_location("tessera_v2_upstream_reference", _REFERENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream():
    return _load_reference_module()


@pytest.fixture(scope="module")
def upstream_model(upstream):
    """Upstream ``PixelStudent`` built by upstream's own ``load_model``."""
    return upstream.load_model(str(_checkpoint_path()), torch.device("cpu"))


@pytest.fixture(scope="module")
def ported_model():
    """Our model, built through the production builder (strict=True load)."""
    config = InferenceConfig(
        time_window=parse_time_window("June 2025"),
        model_version="v2-large",
        dropout=0.0,
        checkpoint_path=str(_checkpoint_path()),
    )
    return build_inference_model(config, torch.device("cpu"))


def _synthetic_batch(
    upstream_module,
    *,
    batch: int,
    t_s2: int,
    t_s1: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardised (S2, S1) inputs shaped like the sampler's output.

    Raw reflectance / backscatter are drawn in plausible ranges and standardised
    with the v2 stats, then a raw integer DOY in [1, 365] is appended as the last
    channel — exactly the contract ``sampling.py`` produces.
    """
    gen = torch.Generator().manual_seed(seed)
    stats = band_stats("v2-large")
    assert stats["s2_mean"] == pytest.approx(upstream_module.S2_BAND_MEAN.tolist())
    assert stats["s2_std"] == pytest.approx(upstream_module.S2_BAND_STD.tolist())
    assert stats["s1_asc_mean"] == pytest.approx(upstream_module.S1A_BAND_MEAN.tolist())
    assert stats["s1_desc_std"] == pytest.approx(upstream_module.S1D_BAND_STD.tolist())

    s2_mean = torch.tensor(stats["s2_mean"])
    s2_std = torch.tensor(stats["s2_std"])
    s1_mean = torch.tensor(stats["s1_asc_mean"])
    s1_std = torch.tensor(stats["s1_asc_std"])

    s2_raw = torch.randint(0, 10000, (batch, t_s2, 10), generator=gen).float()
    s1_raw = torch.randint(0, 12000, (batch, t_s1, 2), generator=gen).float()

    s2 = torch.empty(batch, t_s2, 11)
    s2[..., :10] = (s2_raw - s2_mean) / (s2_std + 1e-9)
    s2[..., 10] = torch.randint(1, 366, (batch, t_s2), generator=gen).float()

    s1 = torch.empty(batch, t_s1, 3)
    s1[..., :2] = (s1_raw - s1_mean) / (s1_std + 1e-9)
    s1[..., 2] = torch.randint(1, 366, (batch, t_s1), generator=gen).float()
    return s2, s1


def test_state_dict_keys_are_identical(ported_model, upstream_model) -> None:
    """Both graphs expose exactly the same parameter names and shapes."""
    ours = {k: tuple(v.shape) for k, v in ported_model.state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in upstream_model.state_dict().items()}
    assert ours == theirs


def test_weights_loaded_are_identical(ported_model, upstream_model) -> None:
    """strict=True load put the same tensors in the same places."""
    theirs = upstream_model.state_dict()
    for key, value in ported_model.state_dict().items():
        torch.testing.assert_close(value, theirs[key], atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("batch", "t_s2", "t_s1"),
    [
        (4, 8, 8),  # smallest bucket
        (6, 16, 24),  # mismatched S2/S1 bucket lengths
        (3, 1, 1),  # single-observation edge case (AttentionPooling passthrough)
        (2, 256, 256),  # largest bucket in the schedule
    ],
)
def test_forward_matches_upstream(ported_model, upstream_model, upstream, batch, t_s2, t_s1) -> None:
    """Our forward equals upstream's ``encode`` on identical synthetic inputs."""
    s2, s1 = _synthetic_batch(upstream, batch=batch, t_s2=t_s2, t_s1=t_s1, seed=t_s2 * 100 + t_s1)

    with torch.no_grad():
        ours = ported_model(s2, s1)
        theirs = upstream_model.encode(s2, s1)

    assert ours.shape == (batch, 128) == theirs.shape
    max_abs_diff = (ours - theirs).abs().max().item()
    assert max_abs_diff <= _ATOL, f"max abs diff {max_abs_diff:.3e} > {_ATOL:.1e}"
    torch.testing.assert_close(ours, theirs, atol=_ATOL, rtol=1e-5)


def test_upstream_output_is_layer_normalised(upstream_model, upstream) -> None:
    """Sanity check on the reference itself: the final LayerNorm is really there."""
    s2, s1 = _synthetic_batch(upstream, batch=5, t_s2=16, t_s1=16, seed=7)
    with torch.no_grad():
        out = upstream_model.encode(s2, s1)
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(5), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.std(dim=-1, unbiased=False), torch.ones(5), atol=1e-4, rtol=0)
