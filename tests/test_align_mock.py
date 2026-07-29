from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sub_align.device import default_compute_type, resolve_device
from sub_align.formats import srt


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_default_compute_type():
    assert default_compute_type("cuda") == "float16"
    assert default_compute_type("cpu") == "int8"
    assert default_compute_type("cpu", "float32") == "float32"


def test_align_file_mocked(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,000 --> 00:00:02,000
World
"""
        ),
    )
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 3, dtype=np.float32)
    fake_audio[16_000:32_000] = 0.2

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {
        "segments": [
            {"text": "Hello", "start": 1.0, "end": 1.4},
            {"text": "World", "start": 1.5, "end": 1.9},
        ]
    }

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        from sub_align.align import align_file

        result = align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            mode="realign",
            margin=0.1,
            vad_method="energy",
        )

    assert result == out
    assert fake_wx.align.called
    cues = srt.load(out)
    assert abs(cues[0].start - 1.0) < 1e-6
    assert abs(cues[1].end - 1.9) < 1e-6
    assert cues[0].text == "Hello"


def test_align_requires_language(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"x")
    subtitle = tmp_path / "a.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="language"):
        align_file(media=media, subtitle=subtitle, detect_language=False)


def test_apply_aligned_times_merges_split_sentences():
    """WhisperX splits multi-sentence cues; times must remap onto original cues."""
    from sub_align.align import _apply_aligned_times
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello, Daniel speaking. How may I help you?", 3.14, 5.94),
        Cue(2, "Hi Daniel. Julie here.", 5.94, 9.78),
    ]
    # WhisperX-style split: 2 cues -> 4 sentence segments
    aligned = [
        {
            "text": "Hello, Daniel speaking.",
            "start": 3.26,
            "end": 4.14,
            "words": [
                {"word": "Hello,", "start": 3.26, "end": 3.54},
                {"word": "Daniel", "start": 3.58, "end": 3.82},
                {"word": "speaking.", "start": 3.84, "end": 4.14},
            ],
        },
        {
            "text": "How may I help you?",
            "start": 4.18,
            "end": 4.72,
            "words": [
                {"word": "How", "start": 4.18, "end": 4.30},
                {"word": "may", "start": 4.32, "end": 4.38},
                {"word": "I", "start": 4.40, "end": 4.42},
                {"word": "help", "start": 4.44, "end": 4.58},
                {"word": "you?", "start": 4.62, "end": 4.72},
            ],
        },
        {
            "text": "Hi Daniel.",
            "start": 5.66,
            "end": 6.20,
            "words": [
                {"word": "Hi", "start": 5.66, "end": 5.80},
                {"word": "Daniel.", "start": 5.84, "end": 6.20},
            ],
        },
        {
            "text": "Julie here.",
            "start": 6.70,
            "end": 7.26,
            "words": [
                {"word": "Julie", "start": 6.70, "end": 6.98},
                {"word": "here.", "start": 7.04, "end": 7.26},
            ],
        },
    ]

    updated = _apply_aligned_times(cues, aligned)
    assert len(updated) == 2
    assert updated[0].text == cues[0].text
    assert abs(updated[0].start - 3.26) < 1e-6
    assert abs(updated[0].end - 4.72) < 1e-6
    assert updated[1].text == cues[1].text
    assert abs(updated[1].start - 5.66) < 1e-6
    assert abs(updated[1].end - 7.26) < 1e-6


def test_apply_aligned_times_handles_contractions():
    from sub_align.align import _apply_aligned_times
    from sub_align.models import Cue

    cues = [Cue(1, "I'm fine.", 1.0, 2.0)]
    aligned = [
        {
            "text": "I'm fine.",
            "start": 1.1,
            "end": 1.8,
            "words": [
                {"word": "I'm", "start": 1.1, "end": 1.3},
                {"word": "fine.", "start": 1.5, "end": 1.8},
            ],
        }
    ]
    updated = _apply_aligned_times(cues, aligned)
    assert abs(updated[0].start - 1.1) < 1e-6
    assert abs(updated[0].end - 1.8) < 1e-6


def test_apply_aligned_times_one_to_one_without_words():
    from sub_align.align import _apply_aligned_times
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello", 0.0, 1.0),
        Cue(2, "World", 1.0, 2.0),
    ]
    aligned = [
        {"text": "Hello", "start": 0.5, "end": 0.9},
        {"text": "World", "start": 1.2, "end": 1.8},
    ]
    updated = _apply_aligned_times(cues, aligned)
    assert abs(updated[0].start - 0.5) < 1e-6
    assert abs(updated[1].end - 1.8) < 1e-6


def test_trim_overlaps_shortens_earlier_cue_only():
    from sub_align.align import _trim_overlaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "A", 11.932, 14.517),
        Cue(2, "B", 14.260, 16.966),
        Cue(3, "C", 18.0, 19.0),
    ]
    updated = _trim_overlaps(cues)
    assert abs(updated[0].start - 11.932) < 1e-6
    assert abs(updated[0].end - 14.260) < 1e-6
    assert abs(updated[1].start - 14.260) < 1e-6
    assert abs(updated[1].end - 16.966) < 1e-6
    assert updated[2] is cues[2]


def test_trim_overlaps_clamps_when_start_past_next():
    from sub_align.align import _trim_overlaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "A", 5.0, 6.0),
        Cue(2, "B", 4.0, 7.0),
    ]
    updated = _trim_overlaps(cues)
    assert abs(updated[0].start - 5.0) < 1e-6
    assert abs(updated[0].end - 5.0) < 1e-6
    assert updated[1] is cues[1]
