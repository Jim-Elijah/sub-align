from __future__ import annotations

import numpy as np
import pytest

from sub_align.models import Cue
from sub_align.vad import energy_speech_spans
from sub_align.windows import assign_windows


def _cues(*texts: str) -> list[Cue]:
    return [
        Cue(index=i + 1, text=text, start=float(i), end=float(i) + 0.5)
        for i, text in enumerate(texts)
    ]


def test_refine_expands_margin():
    cues = _cues("a", "bb")
    segments = assign_windows(cues, mode="refine", margin=0.25, audio_duration=10.0)
    assert segments[0]["start"] == 0.0  # clamped
    assert abs(segments[0]["end"] - 0.75) < 1e-6
    assert abs(segments[1]["start"] - 0.75) < 1e-6


def test_refine_start_margin_asymmetric():
    cues = [
        Cue(index=1, text="hello", start=2.0, end=3.0),
    ]
    segments = assign_windows(
        cues,
        mode="refine",
        margin=0.5,
        start_margin=0.25,
        audio_duration=10.0,
    )
    assert segments[0]["start"] == pytest.approx(1.75)
    assert segments[0]["end"] == pytest.approx(3.5)


def test_realign_skips_leading_silence():
    cues = _cues("hello", "world!!")
    # speech only in the middle of a longer timeline
    spans = [(5.0, 9.0)]
    segments = assign_windows(
        cues,
        mode="realign",
        margin=0.0,
        speech_spans=spans,
        audio_duration=12.0,
    )
    assert segments[0]["start"] >= 5.0
    assert segments[-1]["end"] <= 9.0
    # longer text gets a larger share
    first_dur = segments[0]["end"] - segments[0]["start"]
    second_dur = segments[1]["end"] - segments[1]["start"]
    assert second_dur > first_dur


def test_energy_vad_finds_tone_in_silence():
    sr = 16_000
    silence = np.zeros(sr, dtype=np.float32)  # 1s
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    trailing = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([silence, tone, trailing])
    spans = energy_speech_spans(audio, sample_rate=sr)
    assert spans
    start, end = spans[0]
    assert start >= 0.8
    assert end <= 2.2
