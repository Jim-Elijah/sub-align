from __future__ import annotations

from pathlib import Path

import numpy as np

from sub_align.vad import SAMPLE_RATE


def load_audio(path: str | Path) -> np.ndarray:
    """Load mono 16 kHz float32 audio via WhisperX (requires ffmpeg)."""
    try:
        import whisperx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "whisperx is required to load media. Install with: pip install 'sub-align[align]'"
        ) from exc

    return whisperx.load_audio(str(path))


def audio_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    if audio.size == 0:
        return 0.0
    wave = audio if audio.ndim == 1 else audio.reshape(-1)
    return float(wave.size / sample_rate)


def trim_audio(
    audio: np.ndarray,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Drop leading/trailing samples so alignment ignores intro/outro speech.

    Times are in seconds on the original media timeline. Returns a contiguous
    mono (or flattened) float array; empty result raises ``ValueError``.
    """
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start and trim_end must be >= 0")
    if trim_start == 0 and trim_end == 0:
        return audio

    wave = audio if audio.ndim == 1 else audio.reshape(-1)
    duration = float(wave.size / sample_rate)
    if trim_start + trim_end >= duration:
        raise ValueError(
            f"trim_start ({trim_start}) + trim_end ({trim_end}) "
            f"must be less than audio duration ({duration:.3f}s)"
        )

    start_i = int(round(trim_start * sample_rate))
    end_i = wave.size - int(round(trim_end * sample_rate))
    trimmed = wave[start_i:end_i]
    if trimmed.size == 0:
        raise ValueError("trim left no audio samples")
    return trimmed.astype(audio.dtype, copy=False)
