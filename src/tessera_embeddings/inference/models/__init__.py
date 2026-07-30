"""Tessera dual-Transformer models for multimodal satellite embedding inference."""

from .builder import build_inference_model, load_v2_checkpoint, load_v11_checkpoint
from .modules import TemporalAwarePooling, TemporalPositionalEncoder, V11TransformerEncoder
from .ssl_model import MultimodalBTInferenceModel
from .student_v2 import AttentionPooling, StudentTransformerEncoder, build_v2_dim_reducer

__all__ = [
    "AttentionPooling",
    "MultimodalBTInferenceModel",
    "StudentTransformerEncoder",
    "TemporalAwarePooling",
    "TemporalPositionalEncoder",
    "V11TransformerEncoder",
    "build_inference_model",
    "build_v2_dim_reducer",
    "load_v2_checkpoint",
    "load_v11_checkpoint",
]
