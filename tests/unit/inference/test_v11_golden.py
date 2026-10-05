"""Golden reference test: our v1.1 model == upstream's, on the real checkpoint.

The only reliable proof that our port computes what the model was trained to compute is running
the *upstream* implementation side by side on the same weights and inputs.
``tests/fixtures/upstream/v1_1_reference/`` holds verbatim copies of upstream's v1.1 inference
model; this test loads the real checkpoint into both and asserts the outputs match at FP32 on CPU.

The 231 MB checkpoint is not in git, so the whole module skips unless ``TESSERA_V11_CKPT`` points
at ``tessera_v1_1_aws_encoder.pt``. ADR 026 records what this test would have caught: production
once swapped the pooling GRU for ``nn.GRU``, a different formula.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.models.builder import build_inference_model
from tests._paths import FIXTURES

_REFERENCE_DIR = FIXTURES / "upstream" / "v1_1_reference"

# Observed on the real checkpoint (CPU, FP32): the two graphs agree to FP32 rounding, the fused pooling
# step accounting for all of it (relative 4.4e-7 in the pooled vector). The tolerance absorbs that and
# cross-platform threading, not a structural divergence; the old nn.GRU swap misses by orders more.
_ATOL = 1e-5

# ``tessera_v1_1_aws_encoder.pt`` as production stages it. Verified BEFORE any loader sees the file:
# the vendored upstream loader calls ``torch.load(..., weights_only=False)``, the arbitrary-object
# unpickler, and TESSERA_V11_CKPT is a path the developer supplies.
_CHECKPOINT_SHA256 = "576fbd28f6367f7265db8b3ff85ff8cb8cfe7a8fc9b45a36873ea47ecdfa6e14"
_CHECKPOINT_SIZE_BYTES = 230893213


def _checkpoint_path() -> Path | None:
    """Local v1.1 checkpoint from ``TESSERA_V11_CKPT``, if it exists."""
    raw = os.getenv("TESSERA_V11_CKPT")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


pytestmark = pytest.mark.skipif(
    _checkpoint_path() is None,
    reason="set TESSERA_V11_CKPT to tessera_v1_1_aws_encoder.pt to run",
)


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    """The checkpoint path, refused unless it matches the recorded digest."""
    path = _checkpoint_path()
    assert path is not None  # guaranteed by pytestmark
    if path.stat().st_size != _CHECKPOINT_SIZE_BYTES:
        pytest.fail(f"TESSERA_V11_CKPT is {path.stat().st_size} bytes, expected {_CHECKPOINT_SIZE_BYTES}: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != _CHECKPOINT_SHA256:
        pytest.fail(
            f"TESSERA_V11_CKPT sha256 {digest.hexdigest()} != {_CHECKPOINT_SHA256}. Refusing to unpickle it: {path}"
        )
    return path


@pytest.fixture(scope="module")
def upstream():
    """The vendored upstream package, imported by path (tests/fixtures is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "tessera_v11_upstream_reference",
        _REFERENCE_DIR / "__init__.py",
        submodule_search_locations=[str(_REFERENCE_DIR)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{spec.name}.ssl_model_v1_1")


@pytest.fixture(scope="module")
def upstream_model(upstream, checkpoint):
    """Upstream's model, built and loaded by upstream's own functions from the checkpoint's stored config."""
    config = dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["config"])
    config["num_obs_checkpoints"] = list(range(8, 257, 8))
    model = upstream.build_v1_1_inference_model(config, torch.device("cpu"))
    upstream.load_v1_1_checkpoint(model, str(checkpoint))
    return model.eval()


@pytest.fixture(scope="module")
def ported_model(checkpoint):
    """Our model, built through the production builder."""
    config = InferenceConfig(time_window=parse_time_window("October 2025"), checkpoint_path=str(checkpoint))
    return build_inference_model(config, torch.device("cpu"))


def _batch(batch: int, t_s2: int, t_s1: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardised bands plus a raw integer day of year as the last channel, the sampler's contract."""
    gen = torch.Generator().manual_seed(seed)
    s2 = torch.cat(
        [torch.randn(batch, t_s2, 10, generator=gen), torch.randint(1, 366, (batch, t_s2, 1), generator=gen).float()],
        -1,
    )
    s1 = torch.cat(
        [torch.randn(batch, t_s1, 2, generator=gen), torch.randint(1, 366, (batch, t_s1, 1), generator=gen).float()], -1
    )
    return s2, s1


def test_weights_loaded_are_identical(ported_model, upstream_model) -> None:
    """Both loaders put the same tensors under the same names."""
    ours, theirs = ported_model.state_dict(), upstream_model.state_dict()
    assert ours.keys() == theirs.keys()
    for key, value in ours.items():
        torch.testing.assert_close(value, theirs[key], atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("batch", "t_s2", "t_s1"),
    [
        (4, 8, 8),  # smallest bucket
        (6, 16, 24),  # different S2 and S1 lengths
        (5, 40, 33),  # past the pooling step's 32-step input-projection block
        (2, 256, 256),  # largest bucket in the schedule
    ],
)
def test_forward_matches_upstream(ported_model, upstream_model, batch, t_s2, t_s1) -> None:
    """Our forward pass equals upstream's on identical inputs."""
    s2, s1 = _batch(batch, t_s2, t_s1, seed=t_s2 * 1000 + t_s1)
    with torch.no_grad():
        ours, theirs = ported_model(s2, s1), upstream_model(s2, s1)
    assert ours.shape == theirs.shape == (batch, 192)
    torch.testing.assert_close(ours, theirs, atol=_ATOL, rtol=1e-5)
