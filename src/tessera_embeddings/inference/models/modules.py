"""Neural network modules for the Tessera dual-Transformer model.

Ported from tessera_infer/src/models/modules.py with type hints and ruff compliance.
Logic is unchanged from the original implementation.
"""

import logging
import math
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomGRUCell(nn.Module):
    """GRU cell with explicit separate linear layers for each gate.

    Uses 6 separate ``nn.Linear(bias=False)`` layers for input/hidden gate weights
    plus 3 bias parameters. This matches the tessera beta QAT checkpoint layout.
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

    Used to match the tessera beta QAT checkpoint which stores per-gate linear weights
    instead of the fused weight matrices used by ``nn.GRU``.
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


class TemporalAwarePooling(nn.Module):
    """Temporal-aware pooling: custom GRU for temporal context, LayerNorm, then attention pooling.

    Uses ``CustomGRU`` (explicit per-gate weights) to match the tessera beta QAT checkpoint.
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

        x_context, _ = self.temporal_context(x)
        x_context = self.layer_norm(x_context)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            tg1 = time.monotonic()

        w = torch.softmax(self.query(x_context), dim=1)
        result = (w * x).sum(dim=1)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            tg2 = time.monotonic()
            logger.debug(
                "    PROFILE TemporalAwarePooling: "
                "GRU+LN=%.1fms (input dtype=%s, output dtype=%s)  "
                "attn=%.1fms  TOTAL=%.1fms",
                (tg1 - tg0) * 1000,
                x.dtype,
                x_context.dtype,
                (tg2 - tg1) * 1000,
                (tg2 - tg0) * 1000,
            )

        return result


class TemporalEncoding(nn.Module):
    """Learnable temporal encoding from day-of-year values.

    Uses learnable frequency parameters and a linear projection, unlike the fixed
    sinusoidal ``TemporalPositionalEncoder``. Ported from tessera alpha_1.0 for
    forward-compatibility but NOT currently wired into ``TransformerEncoder``.
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

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # CHANGED: Pre-compute div_term as a non-persistent buffer. It depends only on
        # d_model, not on input data, so recomputing it every forward call wastes ~18ms
        # per backbone. Non-persistent (persistent=False) so it's excluded from state_dict
        # and doesn't break checkpoint loading. Still moves with .to(device) / .bfloat16().
        # Stored in FP32 for numerical precision in sin/cos; output is cast to model dtype.
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """Compute positional encoding from day-of-year.

        Args:
            doy: DOY values of shape (B, T).

        Returns:
            Positional encoding of shape (B, T, d_model).
        """
        # CHANGED: Compute in FP32 for numerical precision, then cast output to match
        # the caller's dtype (BF16 when model.bfloat16() is active). Without this cast,
        # the FP32 positional encoding contaminates all downstream ops via PyTorch's
        # automatic dtype promotion (BF16 + FP32 = FP32), causing the entire transformer
        # and GRU to run in FP32 — 7 TFLOPS instead of 20-30 TFLOPS on T4 tensor cores.
        # See ai/debug/inference_throughput_profiling_2026-02-23.md for full analysis.
        position = doy.unsqueeze(-1).float()

        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term.float())
        pe[:, :, 1::2] = torch.cos(position * self.div_term.float())
        return pe.to(doy.dtype)


class TransformerEncoder(nn.Module):
    """Transformer encoder for satellite time series.

    Takes band values + DOY as input. Embeds bands via MLP, adds sinusoidal
    positional encoding from DOY, runs through transformer encoder layers,
    then pools via TemporalAwarePooling.

    The last column of input x is treated as DOY; all preceding columns are bands.
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

        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

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
            x: Input of shape (B, seq_len, band_num + 1) where the last column is DOY.

        Returns:
            Pooled representation of shape (B, latent_dim * 4).
        """
        profile = getattr(self, "_profile", False)

        bands = x[:, :, :-1]
        doy = x[:, :, -1]

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t0 = time.monotonic()

        bands_embedded = self.embedding(bands)

        if profile and x.is_cuda:
            torch.cuda.synchronize()
            t1 = time.monotonic()

        temporal_encoding = self.temporal_encoder(doy)

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
                "  PROFILE TransformerEncoder (band_num=%d, input shape=%s): "
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
