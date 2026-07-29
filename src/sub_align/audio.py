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
            "whisperx is required to load media. Install with: "
            "pip install 'sub-align[align]'"
        ) from exc

    return whisperx.load_audio(str(path))


def audio_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    if audio.size == 0:
        return 0.0
    wave = audio if audio.ndim == 1 else audio.reshape(-1)
    return float(wave.size / sample_rate)
