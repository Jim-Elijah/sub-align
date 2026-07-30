from __future__ import annotations

import numpy as np
import pytest

from sub_align.audio import audio_duration, trim_audio
from sub_align.vad import SAMPLE_RATE


def test_trim_audio_noop():
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    assert trim_audio(audio) is audio


def test_trim_audio_start_and_end():
    # 5s tone; drop 1s head and 1s tail → 3s
    audio = np.arange(SAMPLE_RATE * 5, dtype=np.float32)
    trimmed = trim_audio(audio, trim_start=1.0, trim_end=1.0)
    assert abs(audio_duration(trimmed) - 3.0) < 1e-6
    assert trimmed[0] == audio[SAMPLE_RATE]
    assert trimmed[-1] == audio[SAMPLE_RATE * 4 - 1]


def test_trim_audio_rejects_negative():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with pytest.raises(ValueError, match=">="):
        trim_audio(audio, trim_start=-0.1)


def test_trim_audio_rejects_over_trim():
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    with pytest.raises(ValueError, match="less than audio duration"):
        trim_audio(audio, trim_start=1.5, trim_end=1.0)
