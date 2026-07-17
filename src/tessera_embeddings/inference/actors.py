"""Ray actor for distributed GPU inference.

Each actor loads the model once, then processes multiple spatial chunks sequentially.
This amortizes model loading and GPU warm-up across chunks.

NOTE: torch and modules that import torch (models.builder, dataset, inference) are
deferred to __init__ / method bodies so the module can be imported on the Fargate
flow runner (which has ray but not torch). Ray serializes the actor class reference;
torch is only needed on GPU workers at runtime.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import ray
import requests

from tessera_embeddings.config.inference import EMBEDDING_DIM, S2_BAND_ORDER, InferenceConfig
from tessera_embeddings.inference.assembly import OBS_COUNT_VARS, ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.data_loading import load_chunk, load_s2_mask_bundle, make_store_opener
from tessera_embeddings.inference.resource_monitor import ResourceMonitor
from tessera_embeddings.storage.zarr_store import credentials_provider

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

    import icechunk
    import torch

    from tessera_embeddings.inference.data_loading import ChunkData, S2MaskBundle, StoreOpener
    from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
    from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel


@dataclass
class _PrefetchedChunk:
    """A chunk prologue loaded ahead of its ``process_chunk`` call.

    Produced by the actor's chunk-prefetch thread while the GPU works on the
    *previous* chunk: everything ``_process_chunk`` needs before its first
    forward pass — repo handles, the full-chunk SCL mask, strip tiling, and
    the first strip's bands + bucketed dataset. Consuming it turns the
    per-chunk GPU-idle prologue (mask read + strip-0 read + dataset build,
    ~45-60 s on production chunks) into work hidden behind the prior chunk's
    inference.
    """

    store_opener: StoreOpener
    mask_bundle: S2MaskBundle
    strip_h: int
    strips: list[slice]
    first_strip: tuple[ChunkData, MosaicChunkInferenceDataset]

logger = logging.getLogger(__name__)

# Host-RAM budget (bytes) for the resident S2 *input* working set of a single
# strip. The 1-deep strip prefetch keeps two band strips resident at once (plus
# the shared full-chunk SCL mask, charged against this budget too), so the real
# ceiling is on the *pair*: 7.0 GB/strip => a 14 GB pair. The remaining headroom
# on the 32 GB g6e.xlarge worker box must absorb the non-S2 resident set the
# sizer does not model — the SAR stack (no cheap pre-read count, so it can spike
# on dense-S1 chunks), the whole-chunk int8 embedding + scale buffers, and the
# model (~4 GB combined on the densest chunks). 7.0 GB/strip raises the prior
# 4.0 GB ceiling with the move from the 16 GB g5.xlarge box to the 32 GB
# g6e.xlarge, but stops short of a full 2x: 4.0 GB was already OOMing on the
# 16 GB box, so the doubled-box budget keeps a deliberate ~4 GB cushion.
_S2_STRIP_BYTE_BUDGET = int(7.0 * 1024**3)

# Per-(timestep, pixel) byte cost of resident S2 bands: 10 bands x uint16.
_S2_BYTES_PER_OBS_PX = len(S2_BAND_ORDER) * 2

# Floor on derived strip height. Below this a strip reads so few northing rows
# that per-strip fixed overhead (zarr open, SCL slice, dataset bucketing)
# dominates and read amplification climbs without meaningfully lowering peak RAM.
# A pathologically dense chunk bottoms out here — deliberately breaching the byte
# budget (logged) — rather than degenerating into hundreds of tiny reads.
_MIN_STRIP_H = 256


def _strip_height_for_density(t_kept: int, width: int, height: int) -> int:
    """Largest northing strip height (rows) whose resident S2 working set fits budget.

    Budget model: at most two S2 band sets are ever co-resident, and each is
    paired with a full-chunk SCL mask (``t_kept * height * width`` bytes, bool).
    The pairing arises two ways, and the sizing must bound the worse of them:

    - *Intra-chunk* (a split chunk's 1-deep strip prefetch): two strips of the
      SAME chunk share ONE mask.
    - *Cross-chunk* (Phase-1 prologue prefetch): this chunk's final strip and
      the NEXT chunk's strip-0 each carry their OWN mask (plus the next chunk's
      SAR, charged to headroom).

    So we require every resident ``bands(strip_h) + full mask <= budget``; two
    such sets then fit ``2 * budget`` in either pairing. This is slightly
    conservative for the intra-chunk case (its mask is shared, not doubled) but
    masks are 1/20th the size of bands, so the over-charge is negligible — and
    charging only half a mask is exactly what OOM-killed a worker: two dense
    (T_kept~120) chunks co-resident under cross-chunk prefetch put ~27 GB on a
    31 GB node (2026-07-17, chunk_5_9, Ray memory monitor kill at 95%).

    A single full-height strip that already fits one budget returns ``height``
    outright. ``t_kept`` is the chunk's true post-prune valid-timestep count, so
    sparse chunks get tall strips (often the whole chunk) and only dense chunks
    split. A chunk dense enough to drive the height below ``_MIN_STRIP_H``
    bottoms out there and breaches the budget (logged) rather than degenerating
    into tiny high-overhead reads.
    """
    t = max(1, t_kept)
    # Full-chunk SCL mask (bool, 1 byte/px), resident the whole loop. Charged in
    # FULL per band set: the cross-chunk pair carries two distinct masks.
    mask_bytes = t * height * width
    # Fast path: the whole chunk (bands + its mask) fits one budget — single
    # strip. Held to ONE budget so a cross-chunk co-load of the next chunk's
    # strip-0 (its own bands + mask) still fits the pair budget.
    if t * height * width * _S2_BYTES_PER_OBS_PX + mask_bytes <= _S2_STRIP_BYTE_BUDGET:
        return height
    band_budget = _S2_STRIP_BYTE_BUDGET - mask_bytes
    per_row = t * width * _S2_BYTES_PER_OBS_PX
    budget_h = max(0, band_budget) // per_row
    if budget_h < _MIN_STRIP_H:
        # Worst-case resident pair: two floor-height band sets, each with a full
        # mask (the cross-chunk pairing).
        pair_gib = 2 * (_MIN_STRIP_H * per_row + mask_bytes) / 1024**3
        logger.warning(
            "S2 density (T_kept=%d, W=%d, H=%d) drives strip_h=%d below floor "
            "%d; using %d. Resident bands+mask pair ~%.1f GiB exceeds the "
            "budget (2 x %.1f GiB) — raise _S2_STRIP_BYTE_BUDGET or expect high "
            "host RAM.",
            t_kept,
            width,
            height,
            budget_h,
            _MIN_STRIP_H,
            _MIN_STRIP_H,
            pair_gib,
            _S2_STRIP_BYTE_BUDGET / 1024**3,
        )
        return _MIN_STRIP_H
    return budget_h


def _strip_slices(height: int, strip_h: int) -> list[slice]:
    """Tile ``[0, height)`` into chunk-relative northing strips of ``strip_h`` rows.

    The final strip is shorter when ``height`` is not a multiple of ``strip_h``.
    ``strip_h >= height`` yields a single full-height strip.
    """
    return [slice(start, min(start + strip_h, height)) for start in range(0, height, strip_h)]


def _fetch_ec2_instance_id() -> str:
    """Fetch EC2 instance ID from instance metadata (IMDSv2).

    Returns "unknown" if metadata is unavailable (e.g., local development).
    """
    try:
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=2,
        ).text

        return requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text
    except Exception:
        return "unknown"


def _configure_actor_logging() -> None:
    """Configure logging for inference actors.

    Sets root logger to INFO with a standard format, silences noisy third-party
    loggers, and enables DEBUG for inference profiling modules.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    for noisy in ("zarr", "zarr.group", "icechunk", "botocore", "s3transfer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Derive from this module's package so the names survive renames — these
    # were once hardcoded to a stale "src.inference.*" prefix, which silently
    # disabled every DEBUG diagnostic (TIMING, EFFECTIVE TFLOPS, autocast probe).
    pkg = __name__.rsplit(".", 1)[0]  # tessera_embeddings.inference
    for name in (
        f"{pkg}.inference",
        f"{pkg}.profiling",
        f"{pkg}.models.modules",
        f"{pkg}.models.ssl_model",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _select_device(torch_mod: types.ModuleType, instance_id: str) -> torch.device:
    """Detect CUDA availability and return the appropriate torch device.

    Args:
        torch_mod: The torch module (passed to avoid top-level import).
        instance_id: EC2 instance ID for log messages.
    """
    if torch_mod.cuda.is_available():
        device = torch_mod.device("cuda")
        gpu_name = torch_mod.cuda.get_device_name(0)
        logger.info("InferenceActor on instance %s, GPU: %s", instance_id, gpu_name)
    else:
        device = torch_mod.device("cpu")
        logger.warning("InferenceActor on instance %s: no GPU available, using CPU", instance_id)
    return device


def _log_vram_breakdown(model: MultimodalBTInferenceModel, torch_mod: types.ModuleType) -> None:
    """Log VRAM usage breakdown after model loading."""
    allocated = torch_mod.cuda.memory_allocated() / 1024**3
    reserved = torch_mod.cuda.memory_reserved() / 1024**3
    total = torch_mod.cuda.get_device_properties(0).total_memory / 1024**3
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    logger.info(
        "VRAM after model load: allocated=%.2f GB, reserved=%.2f GB, total=%.2f GB, "
        "model params=%.2f GB, model buffers=%.2f GB",
        allocated,
        reserved,
        total,
        param_bytes / 1024**3,
        buffer_bytes / 1024**3,
    )


# Host-RAM env vars, all set via runtime_env (not the decorator process) so Ray
# exports them into the worker environment BEFORE this module imports torch /
# instantiates the C allocator — each is read once at init and ignored if set
# later. On the decorator so they cover every creation site: the initial pool,
# the ActorPool replacement path (scheduling.py), and the local CPU runner.
#
# CUBLAS_WORKSPACE_CONFIG caps the cuBLAS host-side workspace. The model runs
# two CUDA streams (one per backbone), so the default workspace is reserved
# twice and inflates the per-chunk host-RAM plateau. Pinning it small claws that
# back. cuBLAS reads the var only when the first CUDA handle is created.
#
# MALLOC_ARENA_MAX=2 / MALLOC_TRIM_THRESHOLD_=0 attack glibc heap retention. The
# per-sub-batch prefetch thread churns numpy/torch CPU buffers every ~250ms;
# across multiple glibc arenas the freed regions are held on per-arena free
# lists at the high-water mark instead of returned to the OS, which an smaps
# rollup shows as dirty-anon RSS (much of it THP-backed). Capping arenas to 2
# (main + prefetch) and forcing trim-on-free returns that churn to the OS.
# These are A/B probes: if peak RSS drops, the plateau was retained churn, not
# live working set; if it barely moves, the working set is genuinely resident
# and chunking is the only remaining lever.
@ray.remote(
    runtime_env={
        "env_vars": {
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "0",
        }
    }
)
class InferenceActor:
    """Ray actor that runs embedding inference on a single GPU or CPU.

    Loads the model checkpoint once at initialization, then processes
    chunks on demand via :meth:`process_chunk`.

    Resource reservations (``num_gpus``, ``num_cpus``, ...) are NOT set on
    the decorator. Callers pass them via ``InferenceActor.options(...)``
    at ``.remote()`` call time so the same class supports GPU and CPU
    deployments without duplication. Typical patterns::

        # GPU worker (production / CUDA host)
        InferenceActor.options(num_gpus=1).remote(config, ckpt)

        # CPU-only worker (local runner, smoke tests)
        InferenceActor.options(num_gpus=0).remote(config, ckpt)

    The optional ``get_credentials`` callback is the icechunk S3 credential
    provider used for every store open in :meth:`process_chunk`. It is injected
    by the cloud-aware caller (the AWS provider passes
    ``iam_icechunk_credentials``) so this domain actor stays free of any
    cloud-SDK import. When ``None``, icechunk falls back to its default AWS
    credential chain — fine for local/moto runs, but on long-lived cloud
    workers that chain can fail to refresh the instance-profile token; see
    ``providers.aws.credentials.iam_icechunk_credentials``.
    """

    def __init__(
        self,
        config: InferenceConfig,
        checkpoint_path: str,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    ) -> None:
        """Initialize actor: download checkpoint (if S3) and load model onto GPU.

        Args:
            config: Inference configuration.
            checkpoint_path: S3 URI or local path to the model checkpoint.
            get_credentials: Optional icechunk S3 credential provider applied
                (via :func:`credentials_provider`) for the duration of every
                :meth:`process_chunk` call. Injected by the cloud-aware caller;
                ``None`` uses icechunk's default credential chain.
        """
        import torch as _torch

        from tessera_embeddings.inference.models.builder import build_inference_model

        _configure_actor_logging()

        self.config = config
        self._get_credentials = get_credentials
        self.instance_id = _fetch_ec2_instance_id()
        self.device = _torch.device("cpu") if self.config.num_gpus == 0 else _select_device(_torch, self.instance_id)

        local_ckpt = download_checkpoint(checkpoint_path) if _is_remote_uri(checkpoint_path) else checkpoint_path
        self.model: MultimodalBTInferenceModel = build_inference_model(
            config,
            self.device,
            checkpoint_path=local_ckpt,
        )

        if self.device.type == "cuda":  # No-op on CPU
            _log_vram_breakdown(self.model, _torch)

        self._resource_monitor = ResourceMonitor(interval_sec=30)
        self._resource_monitor.start()
        logger.info("InferenceActor ready on instance %s", self.instance_id)

    # ------------------------------------------------------------------
    # Cross-chunk prologue prefetch
    # ------------------------------------------------------------------
    # One persistent single-slot thread loads the NEXT chunk's prologue while
    # the GPU runs the current chunk (triggered at the final strip, where only
    # one band strip is resident, so the pair stays within the strip budget).
    # The stash survives across process_chunk calls; the next call consumes it
    # by label. Lazily initialised so test harnesses that build bare actors via
    # object.__new__ keep working without touching __init__.

    def _prefetch_state(self) -> tuple[ThreadPoolExecutor, dict[str, Future[_PrefetchedChunk]]]:
        pool = getattr(self, "_chunk_prefetch_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-prefetch")
            self._chunk_prefetch_pool = pool
            self._prefetched: dict[str, Future[_PrefetchedChunk]] = {}
        return self._chunk_prefetch_pool, self._prefetched

    def _load_chunk_prologue(self, chunk: ChunkSpec, mosaic_base: str) -> _PrefetchedChunk:
        """Load everything _process_chunk needs before its first forward pass.

        Runs either inline (cold start, prefetch miss) or on the chunk-prefetch
        thread. Store opens normally land inside the calling process_chunk's
        scoped credential provider; a prefetch that outlives its originating
        call may open through icechunk's default chain instead — if that fails,
        the consumer falls back to an inline (in-scope) reload, so a
        credential-window miss degrades to the unprefetched behaviour.
        """
        from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset

        store_opener = make_store_opener()
        mask_bundle = load_s2_mask_bundle(mosaic_base, chunk, self.config.time_window, store_opener=store_opener)
        t_kept = int(mask_bundle.mask.shape[0])
        strip_h = _strip_height_for_density(t_kept, chunk.width, chunk.height)
        strips = _strip_slices(chunk.height, strip_h)
        chunk_data = load_chunk(
            chunk,
            mosaic_base,
            time_window=self.config.time_window,
            s1_orbit=self.config.s1_orbit,
            y_sub=strips[0],
            store_opener=store_opener,
            mask_bundle=mask_bundle,
        )
        dataset = MosaicChunkInferenceDataset(
            chunk_data,
            num_obs_checkpoints=self.config.num_obs_checkpoints,
            s1_orbit=self.config.s1_orbit,
        )
        return _PrefetchedChunk(
            store_opener=store_opener,
            mask_bundle=mask_bundle,
            strip_h=strip_h,
            strips=strips,
            first_strip=(chunk_data, dataset),
        )

    def _start_chunk_prefetch(self, chunk: ChunkSpec, mosaic_base: str) -> None:
        """Kick off the next chunk's prologue load on the prefetch thread."""
        pool, stash = self._prefetch_state()
        if chunk.label in stash:
            return
        logger.info("Prefetching next chunk %s prologue in background", chunk.label)
        stash[chunk.label] = pool.submit(self._load_chunk_prologue, chunk, mosaic_base)

    def _take_prefetched(self, label: str) -> _PrefetchedChunk | None:
        """Consume the stashed prologue for ``label``; evict any stale entries.

        Blocks on an in-flight matching prefetch (partial overlap still wins).
        Returns None — falling back to the inline prologue — when there is no
        matching stash or the prefetch failed.
        """
        _, stash = self._prefetch_state()
        for stale in [k for k in stash if k != label]:
            # Reassignment changed our next chunk; drop the stale future and
            # let its (possibly still-loading) result be garbage collected.
            logger.info("Discarding stale prefetched prologue for %s", stale)
            stash.pop(stale).add_done_callback(lambda f: f.exception())
        future = stash.pop(label, None)
        if future is None:
            return None
        try:
            prefetched = future.result()
        except Exception as exc:
            logger.warning("Prefetched prologue for %s failed (%s); reloading inline", label, exc)
            return None
        logger.info("Using prefetched prologue for %s", label)
        return prefetched

    def ping(self) -> bool:
        """No-op health check used to wait for actor initialization.

        Called after actor creation to block until __init__ completes
        (model loaded, GPU ready). Returns True when ready.
        """
        return True

    def get_instance_id(self) -> str:
        """Return the EC2 instance ID this actor is running on."""
        return self.instance_id

    def process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
        prefetch_hint: ChunkSpec | None = None,
    ) -> dict[str, str | int | float]:
        """Process a single spatial chunk: load data, run inference, write output.

        When a ``get_credentials`` provider was injected at construction, reads
        and the staging write run inside a scoped :func:`credentials_provider`
        so every icechunk S3 open resolves through it. The AWS provider passes
        ``iam_icechunk_credentials``, a botocore-backed callback carrying an
        ``expires_after`` so icechunk periodically re-invokes it and botocore
        refreshes the instance-profile token. Without an injected provider,
        icechunk falls back to the Rust AWS SDK's default chain, which resolves
        the instance-profile credential once and — on a long-lived actor — can
        fail to refresh it, failing this chunk and every subsequent one in the
        process with "no providers in chain provided credentials".

        Args:
            chunk: Spatial chunk specification.
            mosaic_base: Base path for the mosaic stores
                (e.g., "s3://bucket/mosaics/small_minnesota").
            staging_base: Base path for staging output.
            run_id: Unique run identifier.
            tracker: Optional ProgressTracker actor handle for batch-level progress.
            prefetch_hint: The chunk the scheduler has reserved as this actor's
                next assignment. Its prologue (mask + strip-0 + dataset) is
                loaded on the chunk-prefetch thread while this chunk's final
                strip runs inference, so the next ``process_chunk`` call starts
                with the GPU-idle prologue already in hand.

        Returns:
            Result dict with chunk label, status, pixel count, and timing.
        """
        cred_scope = (
            credentials_provider(self._get_credentials)
            if self._get_credentials is not None
            else contextlib.nullcontext()
        )
        with cred_scope:
            return self._process_chunk(chunk, mosaic_base, staging_base, run_id, tracker, prefetch_hint)

    def _process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
        prefetch_hint: ChunkSpec | None = None,
    ) -> dict[str, str | int | float]:
        """Run the load → inference → write pipeline for one chunk.

        Always invoked through :meth:`process_chunk`, which establishes the
        scoped icechunk credential provider this body's S3 opens depend on.
        """
        from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
        from tessera_embeddings.inference.inference import run_inference

        t0 = time.monotonic()

        # Report loading phase so stall detection has visibility before batch 50
        if tracker:
            tracker.report.remote(chunk.label, 0, 0, "loading")  # type: ignore[union-attr]

        try:
            # The prologue — repo handles, full-chunk SCL mask (which sizes the
            # density-based strips), and the first strip's bands + dataset — is
            # either consumed from the chunk-prefetch stash (loaded while the
            # PREVIOUS chunk's final strip ran inference) or loaded inline on a
            # cold start / prefetch miss. See _load_chunk_prologue for what it
            # holds and _PrefetchedChunk for why.
            prefetched = self._take_prefetched(chunk.label)
            if prefetched is None:
                prefetched = self._load_chunk_prologue(chunk, mosaic_base)
            store_opener = prefetched.store_opener
            mask_bundle = prefetched.mask_bundle
            strips = prefetched.strips
            # (chunk_data, dataset) for strips[0]; handed to iteration 0 of the
            # strip loop below, then dropped so at most two strips stay resident.
            first_strip: tuple[ChunkData, MosaicChunkInferenceDataset] | None = prefetched.first_strip
            logger.info(
                "Chunk %s: T_kept=%d -> strip_h=%d -> %d strip(s)",
                chunk.label,
                int(mask_bundle.mask.shape[0]),
                prefetched.strip_h,
                len(strips),
            )
            del prefetched

            # Whole-chunk output buffers, allocated once and held for the chunk.
            # We sub-tile INPUTS only; the output and write path are untouched.
            # save_dim mirrors run_inference: the canonical 128-D slice, or the
            # full representation width for smaller (test) models.
            save_dim = min(EMBEDDING_DIM, self.config.representation_dim)
            embeddings = np.zeros((chunk.height, chunk.width, save_dim), dtype=np.int8)
            scales = np.full((chunk.height, chunk.width), np.nan, dtype=np.float32)

            writer = ZarrWriter(staging_base, embedding_dim=save_dim)

            # mask_bundle is passed as an explicit submit() arg rather than
            # captured, so the loop can `del` the only strong reference once the
            # last strip has loaded (a closure capture can't be deleted).
            # Returns (chunk_data, dataset): bucketing runs on the prefetch
            # thread too, so it overlaps the prior strip's GPU work instead of
            # sitting on the critical path between load and inference.
            def _load_strip(y_sub: slice, bundle: S2MaskBundle) -> tuple[ChunkData, MosaicChunkInferenceDataset]:
                data = load_chunk(
                    chunk,
                    mosaic_base,
                    time_window=self.config.time_window,
                    s1_orbit=self.config.s1_orbit,
                    y_sub=y_sub,
                    store_opener=store_opener,
                    mask_bundle=bundle,
                )
                return data, MosaicChunkInferenceDataset(
                    data,
                    num_obs_checkpoints=self.config.num_obs_checkpoints,
                    s1_orbit=self.config.s1_orbit,
                )

            on_batch = (
                (lambda b, t: tracker.report.remote(chunk.label, b, t, "inference"))  # type: ignore[union-attr]
                if tracker
                else None
            )

            # accumulate obs counts per strip into whole-chunk buffers so the
            # single write_chunk carries the full-chunk obs maps.
            obs_buffers: dict[str, np.ndarray] = {
                var: np.zeros((chunk.height, chunk.width), dtype=np.uint16) for var in OBS_COUNT_VARS
            }

            total_valid = 0
            # 1-deep prefetch pipeline: strip i+1 loads (and buckets) while
            # strip i runs inference (same shape as inference.run_inference's
            # prefetcher). Strip 0 arrived already loaded with the prologue.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="strip-prefetch") as pool:
                next_future: Future[tuple[ChunkData, MosaicChunkInferenceDataset]] | None = None

                # Bound to None so the first iteration's `del` has a target; each
                # iteration rebinds both before use.
                chunk_data: ChunkData | None = None
                dataset: MosaicChunkInferenceDataset | None = None
                for i, strip in enumerate(strips):
                    # The previous strip reported "inference"; flip back to
                    # "loading" before blocking on this strip's read so a slow
                    # multi-strip load isn't misclassified as an inference stall
                    # by _poll_tracker. (Strip 0's "loading" was reported above.)
                    if tracker and i > 0:
                        tracker.report.remote(chunk.label, 0, 0, "loading")  # type: ignore[union-attr]

                    # Release the previous strip's input arrays BEFORE blocking
                    # on this strip's load and submitting the next prefetch. The
                    # dataset retains chunk_data.s2_bands (dataset.py), so without
                    # this drop the prior strip's dataset, the current strip, and
                    # the prefetched next strip could all be resident at once —
                    # three strips, though the strip budget is sized for two.
                    del chunk_data, dataset

                    if i == 0:
                        assert first_strip is not None
                        chunk_data, dataset = first_strip
                        first_strip = None
                    else:
                        assert next_future is not None
                        chunk_data, dataset = next_future.result()

                    # Kick off the next strip's load before running inference on this one.
                    next_future = pool.submit(_load_strip, strips[i + 1], mask_bundle) if i + 1 < len(strips) else None

                    if i + 1 == len(strips) and prefetch_hint is not None:
                        # Final strip: only one band strip is resident from here
                        # on, so the next chunk's strip-0 fits the resident pair
                        # budget. Load its prologue while this strip infers (and
                        # through the staging write below).
                        self._start_chunk_prefetch(prefetch_hint, mosaic_base)

                    for var in OBS_COUNT_VARS:
                        arr = getattr(chunk_data, var)
                        if arr is not None:
                            obs_buffers[var][strip] = arr

                    if len(dataset) == 0:
                        # Empty strip: leave its output rows at the initialised
                        # zero embeddings / NaN scale, mirroring run_inference's
                        # handling of fully-invalid chunks and the NaN convention
                        # for "no embedding here" (see #39). The strip still
                        # contributes its (zero) obs counts, already written above.
                        logger.info("Chunk %s strip %s: no valid pixels, leaving zero-filled", chunk.label, strip)
                        continue

                    result = run_inference(self.model, dataset, self.config, self.device, on_batch=on_batch)
                    embeddings[strip] = result.embeddings
                    scales[strip] = result.scales
                    total_valid += len(dataset)

            # All strips done: the last strip's inputs and the full-chunk SCL
            # mask are no longer needed (obs counts and embeddings already live
            # in the whole-chunk output buffers). Free them before the S3 write
            # so peak RAM doesn't carry a dead strip + mask through the write.
            del chunk_data, dataset, mask_bundle

            if total_valid == 0:
                # Every strip was empty. ROI pre-filter means chunks get here
                # only if they intersect the ROI, so this fires only when every
                # pixel in a live chunk fails the SCL/S1 validity thresholds.
                # Assembly will fill the chunk's footprint with zeros/NaN from
                # the Dask graph — no placeholder zarr needed. We still drop a
                # zero-byte skip marker so verify_staged_completeness can
                # distinguish a legitimate skip from a silently-failed chunk
                # (Ray worker crash, etc.).
                logger.info("Chunk %s has no valid pixels, skipping (assembly will fill)", chunk.label)
                writer.write_skip_marker(chunk, run_id)
                return {
                    "chunk": chunk.label,
                    "status": "skipped",
                    "valid_pixels": 0,
                    "elapsed_sec": time.monotonic() - t0,
                    "instance_id": self.instance_id,
                }

            # Report writing phase before S3 write
            if tracker:
                tracker.report.remote(chunk.label, 0, 0, "writing")  # type: ignore[union-attr]

            # Single whole-chunk write — assembly / skip-marker logic untouched.
            writer.write_chunk(
                chunk,
                embeddings,
                run_id,
                embeddings_std=None,
                scales=scales,
                obs_counts=obs_buffers,
            )

            elapsed = time.monotonic() - t0
            logger.info(
                "Chunk %s complete: %d valid pixels, %.1fs",
                chunk.label,
                total_valid,
                elapsed,
            )

            return {
                "chunk": chunk.label,
                "status": "success",
                "valid_pixels": total_valid,
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
            }

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.exception("Chunk %s failed after %.1fs on instance %s", chunk.label, elapsed, self.instance_id)
            return {
                "chunk": chunk.label,
                "status": "failed",
                "error": str(e),
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
            }


# Schemes that mean "fetch this from somewhere else first". A local filesystem
# path (or an explicit file:// URI) is loaded in place by torch.
_REMOTE_CKPT_SCHEMES = ("s3://", "http://", "https://", "gs://", "az://", "abfs://")


def _is_remote_uri(path: str) -> bool:
    """True if ``path`` must be downloaded before torch.load can open it."""
    return path.startswith(_REMOTE_CKPT_SCHEMES)


def _default_checkpoint_cache() -> str:
    """Pick a download cache dir that exists on the running host.

    On AWS DLAMI GPU boxes the NVMe instance store (~1.5 GB/s) is the right
    target — the root EBS volume (~42 MB/s) is too slow and torch.load with
    mmap hangs on it. Off that path (laptops, CI, non-AWS GPUs) the NVMe mount
    doesn't exist, so fall back to a temp dir under the system tmp.
    """
    nvme = Path("/opt/dlami/nvme")
    if nvme.is_dir():
        return str(nvme / "tessera-checkpoints")
    return str(Path(tempfile.gettempdir()) / "tessera-checkpoints")


def download_checkpoint(remote_path: str, local_dir: str | None = None) -> str:
    """Download a model checkpoint from a remote URI to local storage.

    Handles any fsspec-supported remote scheme — ``s3://``, ``https://``
    (e.g. a HuggingFace ``resolve/main`` URL), ``gs://``, etc. The file is
    staged locally because torch.load wants a real path and reads it twice.

    Args:
        remote_path: Remote URI (e.g. ``"s3://bucket/path/model.pt"`` or
            ``"https://huggingface.co/.../tessera_v1_1_aws_encoder.pt"``).
        local_dir: Local directory for downloads. Defaults to the NVMe
            instance store on AWS DLAMI hosts, else a system temp dir.

    Returns:
        Local file path.

    Concurrency: many actors on the same host may call this with the same
    ``remote_path`` and shared cache dir at once (cold cache, 100s of actors).
    The download writes to a unique temp file and is published to the final
    path with an atomic rename, so a concurrent reader never observes a
    partially-written checkpoint and concurrent writers can't corrupt each
    other's output — the last rename wins, and every byte is identical.
    """
    filename = remote_path.rsplit("/", 1)[-1]

    local = Path(local_dir or _default_checkpoint_cache())
    local.mkdir(parents=True, exist_ok=True)
    local_path = local / filename

    if local_path.exists():
        logger.info("Checkpoint already cached: %s", local_path)
        return str(local_path)

    logger.info("Downloading checkpoint: %s → %s", remote_path, local_path)
    # Checkpoints are ~200 MB, so reading the whole file into memory is fine.
    with fsspec.open(remote_path, "rb") as remote:
        data = remote.read()
    # Stage into a unique temp file in the same dir (so the rename stays on one
    # filesystem and is atomic), then atomically publish — concurrent actors
    # publishing the same checkpoint can't observe a half-written file.
    with tempfile.NamedTemporaryFile(dir=local, prefix=f"{filename}.", suffix=".part", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(local_path)

    downloaded_size = local_path.stat().st_size
    logger.info("Download complete: %s (%.1f MB)", local_path, downloaded_size / 1024 / 1024)

    return str(local_path)
