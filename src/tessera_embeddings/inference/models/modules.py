"""Neural network modules for the Tessera dual-Transformer model.

Ported from ``tessera_infer/src/models/modules.py``; logic is unchanged, so the layouts here are
fixed by the checkpoints rather than chosen.
"""

import functools
import logging
import math
import time
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — the conventional spelling

logger = logging.getLogger(__name__)


class CustomGRUCell(nn.Module):
    """GRU cell with explicit separate linear layers for each gate.

    Six ``nn.Linear(bias=False)`` layers for the input/hidden gate weights plus three bias
    parameters, matching the tessera beta QAT checkpoint layout.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input-side weights (no bias on the linears; bias is a separate parameter)
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)

        # Hidden-side weights
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Separate bias parameters per gate
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Compute one GRU timestep.

        Matches the tessera beta QAT training code exactly:
          - Reset gate applied to h_prev BEFORE W_hh multiplication
          - b_h is additive (outside the reset gate product)
          - Update gate: z selects the NEW candidate (not old state)

        Args:
            x_t: Input at current timestep, shape (B, input_size).
            h_prev: Hidden state from previous timestep, shape (B, hidden_size).

        Returns:
            New hidden state of shape (B, hidden_size).
        """
        r = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        z = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        n = torch.tanh(self.W_ih(x_t) + self.W_hh(r * h_prev) + self.b_h)
        return (1 - z) * h_prev + z * n


class CustomGRU(nn.Module):
    """GRU built from ``CustomGRUCell``, matching ``nn.GRU(batch_first=True)`` contract.

    Holds the checkpoint's per-gate weights, which ``TemporalAwarePooling`` reads for its fused
    step. Its own ``forward`` is the plain, step-at-a-time reference that the fused step is tested
    against; inference never calls it.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru_cell = CustomGRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU over a batch-first sequence.

        Args:
            x: Input tensor of shape (B, T, input_size).
            h_0: Optional initial hidden state of shape (1, B, hidden_size).

        Returns:
            Tuple of (outputs, h_t) where outputs has shape (B, T, hidden_size) and
            h_t has shape (1, B, hidden_size).
        """
        batch_size, seq_len, _ = x.shape

        if h_0 is not None:
            h_t = h_0.squeeze(0)
        else:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            outputs[:, t, :] = h_t

        return outputs, h_t.unsqueeze(0)


# Timesteps whose input projections are computed in one matmul: a (steps, B, 3D) block rather than the
# whole sequence, which at T=256 and B=7,168 would be 8 GiB of BF16.
_INPUT_PROJECTION_STEPS = 32


def _gru_pool_step(
    gi: torch.Tensor,
    h: torch.Tensor,
    w_h_rz: torch.Tensor,
    w_hh: torch.Tensor,
    bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    q_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One timestep of the pooling head: the checkpoint's GRU cell, its LayerNorm and its attention score.

    The arithmetic of ``CustomGRUCell``, with the reset gate applied to ``h`` BEFORE ``W_hh`` as the
    model was trained. ``gi`` is this step's input projection for the reset, update and candidate
    gates, ``(B, 3D)``. Gates run in FP32 and the hidden state is kept at ``h``'s dtype.

    Returns the new hidden state and the step's attention score, ``(B,)`` in FP32. The query's bias is
    left out: it is the same at every timestep, so the softmax over time cancels it.
    """
    d = h.shape[-1]
    g = gi.float() + bias
    hf = h.float()
    rz = torch.mm(h, w_h_rz.t()).float()
    r = torch.sigmoid(g[:, :d] + rz[:, :d])
    z = torch.sigmoid(g[:, d : 2 * d] + rz[:, d:])
    n = torch.tanh(g[:, 2 * d :] + torch.mm((r * hf).to(h.dtype), w_hh.t()).float())
    h_new = ((1 - z) * hf + z * n).to(h.dtype)
    score = (F.layer_norm(h_new, (d,), ln_weight, ln_bias).float() * q_weight).sum(-1)
    return h_new, score


@functools.cache
def _compiled_gru_pool_step() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``_gru_pool_step`` compiled into one GPU kernel per step, once per process."""
    return cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], torch.compile(_gru_pool_step))


class TemporalAwarePooling(nn.Module):
    """Temporal-aware pooling: custom GRU for temporal context, LayerNorm, then attention pooling.

    Each timestep's GRU update, LayerNorm and attention score run as one fused step
    (``_gru_pool_step``, compiled on CUDA), so the ``(B, T, D)`` GRU output is never materialised:
    only the hidden state and one score per step are kept. The weights stay in ``CustomGRU``'s
    checkpoint layout, and the result matches running ``CustomGRU``, LayerNorm and attention in turn.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = CustomGRU(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence with temporal context from GRU.

        Args:
            x: Input tensor of shape (B, seq_len, dim).

        Returns:
            Pooled tensor of shape (B, dim).
        """
        profile = getattr(self, "_profile", False)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            tg0 = time.monotonic()

        cell = self.temporal_context.gru_cell
        w_in = torch.cat([cell.W_ir.weight, cell.W_iz.weight, cell.W_ih.weight])
        w_h_rz = torch.cat([cell.W_hr.weight, cell.W_hz.weight])
        bias = torch.cat([cell.b_r, cell.b_z, cell.b_h]).float()
        q_weight = self.query.weight[0].float()
        step = _compiled_gru_pool_step() if x.is_cuda else _gru_pool_step

        h = x.new_zeros(x.shape[0], cell.hidden_size)
        scores = []
        for t0 in range(0, x.shape[1], _INPUT_PROJECTION_STEPS):
            # Time-major, so each step reads one contiguous (B, 3D) slice.
            for gi in torch.matmul(x[:, t0 : t0 + _INPUT_PROJECTION_STEPS].transpose(0, 1), w_in.t()):
                h, score = step(
                    gi, h, w_h_rz, cell.W_hh.weight, bias, self.layer_norm.weight, self.layer_norm.bias, q_weight
                )
                scores.append(score)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            tg1 = time.monotonic()

        w = torch.softmax(torch.stack(scores, dim=1), dim=1).to(x.dtype).unsqueeze(-1)
        result = (w * x).sum(dim=1)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            tg2 = time.monotonic()
            logger.debug(
                "    PROFILE TemporalAwarePooling: GRU+LN=%.1fms (input dtype=%s)  attn=%.1fms  TOTAL=%.1fms",
                (tg1 - tg0) * 1000,
                x.dtype,
                (tg2 - tg1) * 1000,
                (tg2 - tg0) * 1000,
            )

        return result


class TemporalEncoding(nn.Module):
    """Learnable temporal encoding from day-of-year values.

    Learnable frequency parameters and a linear projection, unlike the fixed sinusoidal
    ``TemporalPositionalEncoder``. Kept from tessera alpha_1.0 for forward-compatibility; NOT
    wired into ``V11TransformerEncoder``.
    """

    def __init__(self, d_model: int, num_freqs: int = 64) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model

        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, math.log(365.0), num_freqs)))
        self.proj = nn.Linear(2 * num_freqs, d_model)
        self.phase = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Compute learnable temporal encoding from day-of-year.

        Args:
            doy: DOY values of shape (B, T, 1).

        Returns:
            Temporal encoding of shape (B, T, d_model).
        """
        t = doy / 365.0 * 2 * math.pi

        t_scaled = t * self.freqs.view(1, 1, -1)  # (B, T, num_freqs)
        sin = torch.sin(t_scaled + self.phase[..., : self.num_freqs])
        cos = torch.cos(t_scaled + self.phase[..., self.num_freqs : 2 * self.num_freqs])

        encoding = torch.cat([sin, cos], dim=-1)  # (B, T, 2*num_freqs)
        return self.proj(encoding)  # (B, T, d_model)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding from day-of-year values.

    Uses fixed sinusoidal frequencies (standard transformer positional encoding)
    driven by DOY values instead of integer positions.
    """

    def __init__(self, d_model: int, *, frequencies_on_cpu: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        # Where the frequencies' exp() runs, as each model's upstream runs it: v1.1 computes them on the
        # CPU and moves them to the device, v2 computes them on the device. The two round differently in
        # the last place on 116 of v1.1's 384 frequencies, so each model follows its own upstream.
        self._frequencies_on_cpu = frequencies_on_cpu
        # div_term depends only on d_model, so it is not a per-forward compute (~18 ms per
        # backbone). It is cached per device rather than held as a registered buffer *on purpose*:
        # a buffer is swept by ``nn.Module.bfloat16()``, which would round these frequencies to 8
        # mantissa bits and shift sin/cos arguments by up to ~0.70 rad at DOY 365 (measured over
        # d_model=640; ~0.65 for v1.1's 768) — a silent divergence from the FP32 path the models
        # were trained and golden-tested on. A plain fp32 cache cannot be downcast by a
        # module-level dtype conversion.
        self._div_term_cache: dict[torch.device, torch.Tensor] = {}

    def _div_term(self, device: torch.device) -> torch.Tensor:
        """FP32 sinusoidal frequencies for *device*, computed once per device.

        Filled on whichever CUDA stream runs the first forward (a backbone side stream under the
        pipelined loop) and read from that same stream on every later call, since each backbone
        owns its own encoder. It is never freed — the cache holds the only reference for the
        module's life — so it needs no ``record_stream``, and the profile batch's default-stream
        read is ordered behind the pipeline drain that precedes it.
        """
        cached = self._div_term_cache.get(device)
        if cached is None:
            where = torch.device("cpu") if self._frequencies_on_cpu else device
            cached = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32, device=where)
                * -(math.log(10000.0) / self.d_model)
            ).to(device)
            self._div_term_cache[device] = cached
        return cached

    def forward(self, doy: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Compute positional encoding from day-of-year.

        Args:
            doy: DOY values of shape (B, T). Must be FP32 — see the dtype note
                below; BF16 cannot represent DOY above 256 to one-day resolution.
            out_dtype: dtype of the returned encoding. Defaults to ``doy.dtype``.

        Returns:
            Positional encoding of shape (B, T, d_model).
        """
        # FP32 for precision, then cast the OUTPUT to the compute dtype (BF16 under
        # model.bfloat16()). Without that cast PyTorch's dtype promotion (BF16 + FP32 = FP32)
        # spreads FP32 through the whole transformer and GRU: 7 TFLOPS instead of 20-30 on T4
        # tensor cores.
        #
        # *doy itself must arrive in FP32.* BF16 carries 8 mantissa bits, so it represents integers
        # exactly only up to 256: DOY 257 would land on 256, and .float() here cannot recover what
        # the cast already discarded. The callers therefore keep the DOY channel FP32 and cast only
        # the bands — see V11TransformerEncoder.forward / StudentTransformerEncoder.forward.
        #
        # torch.empty, not zeros: every element is written because 0::2 and 1::2 partition the
        # even d_model, so the multi-GB zero-fill is dead work and the values are identical. The
        # strided writes also avoid holding sin and cos live beside an interleaved output.
        position = doy.float().unsqueeze(-1)
        angles = position * self._div_term(doy.device)  # (B, T, d_model/2)
        pe = torch.empty(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        # Drop angles before the cast: ~2.6 GiB at the largest (B=7168, T=256) bucket and unused
        # from here, so holding it through pe.to() co-resides it with the fp32 pe and the bf16
        # output — peak VRAM the concurrent s2/s1 backbones cannot spare.
        del angles
        return pe.to(out_dtype or doy.dtype)


class V11TransformerEncoder(nn.Module):
    """Transformer encoder for satellite time series.

    Embeds bands via MLP, adds sinusoidal positional encoding from DOY, runs the transformer
    encoder layers, then pools via ``TemporalAwarePooling``. The last column of ``x`` is DOY;
    every preceding column is a band.
    """

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ) -> None:
        super().__init__()
        _ = max_seq_len  # Unused but kept for checkpoint compatibility
        input_dim = band_num

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )

        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4, frequencies_on_cpu=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attn_pool = TemporalAwarePooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, seq_len, band_num + 1) where the last column is
                DOY. Pass FP32 even under reduced-precision inference: the bands
                are cast to the compute dtype here, while the DOY column stays
                FP32 so integer days above 256 survive (see
                :class:`TemporalPositionalEncoder`).

        Returns:
            Pooled representation of shape (B, latent_dim * 4).
        """
        profile = getattr(self, "_profile", False)

        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        # The compute dtype is whatever the weights carry (BF16/FP16 after
        # model.bfloat16()/.half(), FP32 otherwise). Casting per-channel here
        # instead of casting the whole input upstream is what keeps DOY exact;
        # when x already matches, both .to() calls are no-ops.
        compute_dtype = cast("nn.Linear", self.embedding[0]).weight.dtype

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t0 = time.monotonic()

        bands_embedded = self.embedding(bands.to(compute_dtype))

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t1 = time.monotonic()

        temporal_encoding = self.temporal_encoder(doy, out_dtype=compute_dtype)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t2 = time.monotonic()

        x = bands_embedded + temporal_encoding

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t3 = time.monotonic()
            # Capture add output dtype BEFORE x is overwritten by transformer_encoder
            add_dtype = x.dtype

        x = self.transformer_encoder(x)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t4 = time.monotonic()

        result = self.attn_pool(x)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t5 = time.monotonic()
            band_num = bands.shape[-1]
            logger.debug(
                "  PROFILE V11TransformerEncoder (band_num=%d, input shape=%s): "
                "embedding=%.1fms (out dtype=%s)  "
                "pos_encoding=%.1fms (out dtype=%s)  "
                "add=%.1fms (out dtype=%s)  "
                "transformer=%.1fms (out dtype=%s)  "
                "gru+attn_pool=%.1fms (out dtype=%s)  "
                "TOTAL=%.1fms",
                band_num,
                list(bands.shape),
                (t1 - t0) * 1000,
                bands_embedded.dtype,
                (t2 - t1) * 1000,
                temporal_encoding.dtype,
                (t3 - t2) * 1000,
                add_dtype,
                (t4 - t3) * 1000,
                x.dtype,
                (t5 - t4) * 1000,
                result.dtype,
                (t5 - t0) * 1000,
            )

        return result
