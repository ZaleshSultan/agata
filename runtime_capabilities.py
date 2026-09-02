"""
Runtime Capability Detector for Agata.

Determines GPU availability, PyTorch CUDA readiness, CTranslate2 CUDA support,
and provides unified device & compute type resolution for STT (faster-whisper)
and TTS (XTTS-v2).
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemCapabilities:
    has_torch_cuda: bool
    gpu_name: str
    vram_total_mb: float
    vram_free_mb: float
    has_ctranslate2_cuda: bool
    ctranslate2_cuda_compute_types: frozenset[str]
    recommended_stt_device: str
    recommended_stt_compute: str
    recommended_tts_device: str


_CAPABILITIES_CACHE: SystemCapabilities | None = None


def get_system_capabilities(force_refresh: bool = False) -> SystemCapabilities:
    """Detect and return current system capabilities (cached)."""
    global _CAPABILITIES_CACHE
    if _CAPABILITIES_CACHE is not None and not force_refresh:
        return _CAPABILITIES_CACHE

    has_torch_cuda = False
    gpu_name = "N/A"
    vram_total_mb = 0.0
    vram_free_mb = 0.0

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_torch_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_mb = props.total_memory / (1024 * 1024)
            # Free VRAM estimation
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                vram_free_mb = free_b / (1024 * 1024)
            except Exception:
                vram_free_mb = vram_total_mb
    except Exception as exc:
        logger.debug("[CAPABILITIES] PyTorch CUDA check error: %s", exc)

    has_ctranslate2_cuda = False
    ctranslate2_compute_types: set[str] = set()

    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            types = ctranslate2.get_supported_compute_types("cuda")
            if types:
                has_ctranslate2_cuda = True
                ctranslate2_compute_types = set(types)
    except Exception as exc:
        logger.debug("[CAPABILITIES] CTranslate2 CUDA check error: %s", exc)

    # Determine recommended STT settings
    if has_ctranslate2_cuda:
        recommended_stt_device = "cuda"
        if "float16" in ctranslate2_compute_types:
            recommended_stt_compute = "float16"
        elif "int8_float16" in ctranslate2_compute_types:
            recommended_stt_compute = "int8_float16"
        elif "int8" in ctranslate2_compute_types:
            recommended_stt_compute = "int8"
        else:
            recommended_stt_compute = "default"
    else:
        recommended_stt_device = "cpu"
        recommended_stt_compute = "int8"

    # Determine recommended TTS settings
    if has_torch_cuda and vram_total_mb >= 3500:
        recommended_tts_device = "cuda"
    else:
        recommended_tts_device = "cpu"

    caps = SystemCapabilities(
        has_torch_cuda=has_torch_cuda,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        has_ctranslate2_cuda=has_ctranslate2_cuda,
        ctranslate2_cuda_compute_types=frozenset(ctranslate2_compute_types),
        recommended_stt_device=recommended_stt_device,
        recommended_stt_compute=recommended_stt_compute,
        recommended_tts_device=recommended_tts_device,
    )

    logger.info(
        "[CAPABILITIES] GPU='%s' (%.0fMB VRAM) | PyTorch CUDA=%s | CTranslate2 CUDA=%s | STT rec=%s/%s | TTS rec=%s",
        caps.gpu_name,
        caps.vram_total_mb,
        caps.has_torch_cuda,
        caps.has_ctranslate2_cuda,
        caps.recommended_stt_device,
        caps.recommended_stt_compute,
        caps.recommended_tts_device,
    )

    _CAPABILITIES_CACHE = caps
    return caps


def resolve_stt_device(
    requested_device: str = "auto",
    requested_compute: str = "auto",
) -> tuple[str, str]:
    """
    Resolve STT device and compute type based on user config and hardware.

    Returns:
        (device, compute_type) e.g. ("cuda", "float16") or ("cpu", "int8")
    """
    caps = get_system_capabilities(force_refresh=True)
    req_dev = requested_device.lower().strip()
    req_comp = requested_compute.lower().strip()

    if req_dev == "cpu":
        dev = "cpu"
        comp = "int8" if req_comp in ("auto", "default") else req_comp
        return dev, comp

    if req_dev in ("cuda", "gpu"):
        if caps.has_ctranslate2_cuda:
            dev = "cuda"
            comp = caps.recommended_stt_compute if req_comp == "auto" else req_comp
            return dev, comp
        else:
            logger.warning("[CAPABILITIES] Requested CUDA STT but CTranslate2 CUDA unavailable; falling back to CPU")
            return "cpu", "int8"

    # auto mode
    return caps.recommended_stt_device, caps.recommended_stt_compute


def resolve_tts_device(requested_device: str = "auto") -> str:
    """
    Resolve TTS device ("cuda" or "cpu").
    """
    caps = get_system_capabilities(force_refresh=True)
    req_dev = (requested_device or "auto").lower().strip()

    if req_dev == "cpu":
        return "cpu"

    if req_dev in ("cuda", "gpu"):
        if caps.has_torch_cuda:
            return "cuda"
        logger.warning("[CAPABILITIES] Requested CUDA TTS but PyTorch CUDA unavailable; falling back to CPU")
        return "cpu"

    # auto mode
    return caps.recommended_tts_device
