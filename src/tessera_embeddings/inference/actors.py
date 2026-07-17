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
from typing import TYPE_CHECKING, Any

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
class _ChunkPrologue:
    """Everything ``_process_chunk`` needs before its first forward pass.

    Repo handles, the full-chunk SCL mask, strip tiling, and the first strip's
    bands + bucketed dataset — loaded serially (GPU idle) at the top of every
    chunk by :meth:`InferenceActor._load_chunk_prologue`.
    """

    store_opener: StoreOpener
    mask_bundle: S2MaskBundle
    strip_h: int
    strips: list[slice]
    first_strip: tuple[ChunkData, MosaicChunkInferenceDataset]
    # Chunk-relative easting window of the S2 valid-pixel bounding box, or None
    # for full width. Sparse/edge chunks read (and infer) only these columns;
    # outputs are placed at this offset in the whole-chunk buffers.
    x_sub: slice | None


logger = logging.getLogger(__name__)

# Host-RAM budget (bytes) for one resident S2 band set (a strip's bands + its
# full-chunk SCL mask). The intra-chunk 1-deep strip prefetch keeps TWO such
# sets resident at once, so the steady-state S2 ceiling is the pair:
# 4.75 GiB/set => ~9.5 GiB. Sized for UTM-zone-scale runs, where chunk-density
# variance produces RAM spikes and the operating target is peak host RAM in
# the ~50-60% band of the 30.9 GB usable on a g6e.xlarge:
#   pair ~9.5 + SAR ~1.5 (unmodelled; spikes on dense-S1 chunks) + whole-chunk
#   int8 output buffers ~0.6 + model/torch/misc baseline ~5 => ~16-17 GB ~ 54%.
# History: 7.0 GiB/set held ~68-75% peak once cross-chunk prologue prefetch was
# removed (and ~92-95% with it — the chunk_5_9 OOM). Cross-chunk interleaving
# is deliberately gone (see _load_chunk_prologue); do not raise this budget
# without re-deriving the arithmetic above against the RAM target.
_S2_STRIP_BYTE_BUDGET = int(4.75 * 1024**3)

# Per-(timestep, pixel) byte cost of resident S2 bands: 10 bands x uint16.
_S2_BYTES_PER_OBS_PX = len(S2_BAND_ORDER) * 2

# Cores left free for the inference loop's batch-prep workers while a
# BACKGROUND strip load decompresses bands (intra-chunk strip prefetch). The
# prep pool runs 2 workers (inference.PREFETCH_DEPTH); without this reservation
# four decompression threads contend with them on the 4-vCPU g6e.xlarge and
# get_batch spikes to ~500 ms, starving the GPU mid-inference. Foreground
# (prologue) loads reserve nothing — the GPU is idle and wants the fastest load.
_BACKGROUND_LOAD_RESERVED_CPUS = 2

# Apply the S2 easting-bbox crop only when it removes at least this fraction
# of the chunk's width. Near-full boxes (interior chunks) skip the crop
# entirely, keeping the mainline path byte-for-byte identical to the uncropped
# code and avoiding pointless SAR column-copies for a few saved columns.
_X_CROP_MIN_SAVING = 0.10

# Floor on derived strip height. Below this a strip reads so few northing rows
# that per-strip fixed overhead (zarr open, SCL slice, dataset bucketing)
# dominates and read amplification climbs without meaningfully lowering peak RAM.
# A pathologically dense chunk bottoms out here — deliberately breaching the byte
# budget (logged) — rather than degenerating into hundreds of tiny reads.
_MIN_STRIP_H = 256


def _strip_height_for_density(t_kept: int, width: int, height: int) -> int:
    """Largest northing strip height (rows) whose resident S2 working set fits budget.

    Budget model: the intra-chunk 1-deep strip prefetch keeps at most two S2
    band sets co-resident (strip *i* inferring while strip *i+1* loads), and
    the full-chunk SCL mask stays resident throughout. We require every
    resident ``bands(strip_h) + full mask <= budget``, so the steady-state
    pair fits ``2 * budget``. Charging the (shared) mask in full to each set
    is slightly conservative — masks are ~1/20th the size of bands — and buys
    margin against what the budget deliberately does NOT model: the SAR stack
    (which can spike on dense-S1 chunks) and UTM-zone-scale chunk-density
    variance. There is no cross-chunk co-residency to model: cross-chunk
    prologue prefetch was removed (see ``_load_chunk_prologue``).

    A single full-height strip that already fits one budget returns ``height``
    outright. ``t_kept`` is the chunk's true post-prune valid-timestep count, so
    sparse chunks get tall strips (often the whole chunk) and only dense chunks
    split. A chunk dense enough to drive the height below ``_MIN_STRIP_H``
    bottoms out there and breaches the budget (logged) rather than degenerating
    into tiny high-overhead reads.
    """
    t = max(1, t_kept)
    # Full-chunk SCL mask (bool, 1 byte/px), resident the whole loop; charged
    # in full per band set (conservative; see docstring).
    mask_bytes = t * height * width
    # Fast path: the whole chunk (bands + mask) fits one budget — single strip.
    if t * height * width * _S2_BYTES_PER_OBS_PX + mask_bytes <= _S2_STRIP_BYTE_BUDGET:
        return height
    band_budget = _S2_STRIP_BYTE_BUDGET - mask_bytes
    per_row = t * width * _S2_BYTES_PER_OBS_PX
    budget_h = max(0, band_budget) // per_row
    if budget_h < _MIN_STRIP_H:
        # Worst-case resident pair: two floor-height band sets, each charged a
        # full mask (conservative; the intra-chunk pair actually shares one).
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

    def _load_chunk_prologue(self, chunk: ChunkSpec, mosaic_base: str) -> _ChunkPrologue:
        """Load everything _process_chunk needs before its first forward pass.

        Runs inline (serially, GPU idle) at the top of every chunk. This was
        briefly prefetched across chunk boundaries (cross-chunk interleaving,
        PR #85 Phase 1), which hid it behind the previous chunk's inference —
        but co-residing two chunks' input working sets pushed peak host RAM
        to ~92-95% of the node, and at UTM-zone scale chunk-density variance
        makes that an OOM guarantee. Removed 2026-07-17 to hold peak RAM in
        the ~50-60% band; the serial-load GPU idle is the accepted cost.
        """
        from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset

        store_opener = make_store_opener()
        mask_bundle = load_s2_mask_bundle(mosaic_base, chunk, self.config.time_window, store_opener=store_opener)
        t_kept = int(mask_bundle.mask.shape[0])

        # S2 valid-pixel bounding box in easting. On sparse/edge chunks (a
        # coastline sliver, a UTM-zone boundary) the valid columns can be a
        # small fraction of the width — the S2 band read (20 B/px) shrinks to
        # the box. Columns outside it have zero valid S2 observations, so they
        # could never be inferred; the saved obs layers keep full-extent
        # fidelity via the bundle (S2) and full-width SAR reads (see
        # load_chunk's x_sub docs). Near-full boxes skip the crop so interior
        # chunks stay on the byte-identical uncropped path.
        x_sub: slice | None = None
        valid_cols = np.flatnonzero(mask_bundle.mask.any(axis=(0, 1)))
        if valid_cols.size:
            box = slice(int(valid_cols[0]), int(valid_cols[-1]) + 1)
            if (box.stop - box.start) <= (1 - _X_CROP_MIN_SAVING) * chunk.width:
                x_sub = box
                logger.info(
                    "Chunk %s: S2 valid bbox covers columns %d-%d (%.0f%% of width) — cropping reads",
                    chunk.label,
                    box.start,
                    box.stop,
                    100.0 * (box.stop - box.start) / chunk.width,
                )

        effective_width = chunk.width if x_sub is None else (x_sub.stop - x_sub.start)
        strip_h = _strip_height_for_density(t_kept, effective_width, chunk.height)
        strips = _strip_slices(chunk.height, strip_h)
        chunk_data = load_chunk(
            chunk,
            mosaic_base,
            time_window=self.config.time_window,
            s1_orbit=self.config.s1_orbit,
            y_sub=strips[0],
            store_opener=store_opener,
            mask_bundle=mask_bundle,
            x_sub=x_sub,
        )
        dataset = MosaicChunkInferenceDataset(
            chunk_data,
            num_obs_checkpoints=self.config.num_obs_checkpoints,
            s1_orbit=self.config.s1_orbit,
        )
        return _ChunkPrologue(
            store_opener=store_opener,
            mask_bundle=mask_bundle,
            strip_h=strip_h,
            strips=strips,
            first_strip=(chunk_data, dataset),
            x_sub=x_sub,
        )

    # ------------------------------------------------------------------
    # Deferred staging writes
    # ------------------------------------------------------------------
    # A chunk's staging upload (~seconds of pure I/O) runs on a single-slot
    # writer thread so it overlaps the NEXT chunk's serial prologue load
    # instead of adding GPU-idle time. Durability protocol: the deferring
    # result carries ``write_deferred=True`` and is NOT counted complete by
    # the scheduler until the write's outcome arrives — piggybacked as
    # ``prior_write`` on this actor's next result, or via ``flush_writes()``
    # when the actor idles. RAM cost: one chunk's output buffers (~0.6 GB)
    # held until the upload lands. Lazily initialised for bare-actor tests.

    def _writer_pool_handle(self) -> ThreadPoolExecutor:
        pool = getattr(self, "_writer_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="staging-write")
            self._writer_pool = pool
            self._pending_write: tuple[str, Future[str]] | None = None
        return pool

    def _collect_prior_write(self) -> dict[str, Any] | None:
        """Resolve the outstanding deferred write, if any.

        Blocks until the upload finishes — by construction it started at least
        one chunk ago, so it is almost always already done. Returns
        ``{"label", "ok", "error"}`` or ``None`` when nothing was pending.
        """
        self._writer_pool_handle()
        pending = self._pending_write
        if pending is None:
            return None
        label, future = pending
        self._pending_write = None
        try:
            future.result()
        except Exception as exc:
            logger.warning("Deferred staging write for %s FAILED: %s", label, exc)
            return {"label": label, "ok": False, "error": str(exc)}
        return {"label": label, "ok": True, "error": None}

    def flush_writes(self) -> dict[str, Any] | None:
        """Drain the outstanding deferred write (scheduler calls this when the
        actor idles or at end-of-run, when no further result can carry it).
        """
        return self._collect_prior_write()

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
    ) -> dict[str, Any]:
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

        Returns:
            Result dict with chunk label, status, pixel count, and timing.
        """
        cred_scope = (
            credentials_provider(self._get_credentials)
            if self._get_credentials is not None
            else contextlib.nullcontext()
        )
        with cred_scope:
            return self._process_chunk(chunk, mosaic_base, staging_base, run_id, tracker)

    def _process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
    ) -> dict[str, Any]:
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
            # density-based strips), and the first strip's bands + dataset —
            # loads serially here, GPU idle. Deliberately NOT prefetched across
            # chunk boundaries: see _load_chunk_prologue for the UTM-zone-scale host-RAM
            # rationale.
            prologue = self._load_chunk_prologue(chunk, mosaic_base)
            store_opener = prologue.store_opener
            mask_bundle = prologue.mask_bundle
            strips = prologue.strips
            x_sub = prologue.x_sub
            # Column window the cropped grids map to in the whole-chunk output
            # buffers (full width when uncropped).
            cols = x_sub if x_sub is not None else slice(0, chunk.width)
            # (chunk_data, dataset) for strips[0]; handed to iteration 0 of the
            # strip loop below, then dropped so at most two strips stay resident.
            first_strip: tuple[ChunkData, MosaicChunkInferenceDataset] | None = prologue.first_strip
            logger.info(
                "Chunk %s: T_kept=%d -> strip_h=%d -> %d strip(s)",
                chunk.label,
                int(mask_bundle.mask.shape[0]),
                prologue.strip_h,
                len(strips),
            )
            del prologue

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
                # Background load (runs while the GPU infers the prior strip):
                # reserve cores for the batch-prep workers feeding the GPU, so
                # band decompression can't starve inference. The prologue's
                # strip-0 load (_load_chunk_prologue) reserves nothing — it is
                # serial and the GPU is idle, so it wants every core.
                data = load_chunk(
                    chunk,
                    mosaic_base,
                    time_window=self.config.time_window,
                    s1_orbit=self.config.s1_orbit,
                    y_sub=y_sub,
                    store_opener=store_opener,
                    mask_bundle=bundle,
                    reserve_cpus=_BACKGROUND_LOAD_RESERVED_CPUS,
                    x_sub=x_sub,
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

                    if x_sub is None:
                        for var in OBS_COUNT_VARS:
                            arr = getattr(chunk_data, var)
                            if arr is not None:
                                obs_buffers[var][strip] = arr
                    else:
                        # Cropped grid: the saved obs layers keep full-extent
                        # fidelity — S2 counts come from the (full-width) mask
                        # bundle, SAR counts from the full-width side channel
                        # (SAR is read full-width regardless; see load_chunk).
                        obs_buffers["s2_obs_count"][strip] = mask_bundle.obs_count[strip, :]
                        for var, full in (
                            ("s1_asc_obs_count", chunk_data.s1_asc_obs_count_full),
                            ("s1_desc_obs_count", chunk_data.s1_desc_obs_count_full),
                        ):
                            if full is not None:
                                obs_buffers[var][strip] = full

                    if len(dataset) == 0:
                        # Empty strip: leave its output rows at the initialised
                        # zero embeddings / NaN scale, mirroring run_inference's
                        # handling of fully-invalid chunks and the NaN convention
                        # for "no embedding here" (see #39). The strip still
                        # contributes its (zero) obs counts, already written above.
                        logger.info("Chunk %s strip %s: no valid pixels, leaving zero-filled", chunk.label, strip)
                        continue

                    result = run_inference(self.model, dataset, self.config, self.device, on_batch=on_batch)
                    # Cropped grids land at their column offset; outside the
                    # box the buffers keep their initial zero/NaN fill — the
                    # exact values those never-valid pixels get today.
                    embeddings[strip, cols] = result.embeddings
                    scales[strip, cols] = result.scales
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
                writer.write_skip_marker(chunk, run_id)  # zero-byte marker: keep synchronous
                return {
                    "chunk": chunk.label,
                    "status": "skipped",
                    "valid_pixels": 0,
                    "elapsed_sec": time.monotonic() - t0,
                    "instance_id": self.instance_id,
                    "prior_write": self._collect_prior_write(),
                }

            # Resolve the PREVIOUS chunk's deferred write first (it queued a
            # full chunk ago on the single-slot writer — normally long done),
            # so its outcome rides this result back to the scheduler.
            prior_write = self._collect_prior_write()

            # Defer the whole-chunk staging write to the writer thread: it
            # overlaps the next chunk's serial prologue load instead of adding
            # GPU-idle time. This chunk is only counted complete once the
            # write's outcome is confirmed (see class comment above).
            self._writer_pool_handle()
            self._pending_write = (
                chunk.label,
                self._writer_pool.submit(
                    writer.write_chunk,
                    chunk,
                    embeddings,
                    run_id,
                    embeddings_std=None,
                    scales=scales,
                    obs_counts=obs_buffers,
                ),
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
                "write_deferred": True,
                "prior_write": prior_write,
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
