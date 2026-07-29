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
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "Hello", "start": 1.0, "end": 1.4},
            {"text": "World", "start": 1.5, "end": 1.9},
        ]
    }
    fake_wx.load_model.return_value = fake_model

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
            margin=0.1,
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


def test_align_txt_uses_transcription_windows(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.txt"
    subtitle.write_text("Hello\nWorld\n", encoding="utf-8")
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 3, dtype=np.float32)
    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    # 1st align: word-time the ASR; 2nd align: script cues.
    fake_wx.align.side_effect = [
        {
            "segments": [
                {
                    "text": "Hello",
                    "start": 1.0,
                    "end": 1.4,
                    "words": [{"word": "Hello", "start": 1.0, "end": 1.4}],
                },
                {
                    "text": "World",
                    "start": 1.5,
                    "end": 1.9,
                    "words": [{"word": "World", "start": 1.5, "end": 1.9}],
                },
            ]
        },
        {
            "segments": [
                {
                    "text": "Hello World",
                    "start": 1.0,
                    "end": 1.9,
                    "words": [
                        {"word": "Hello", "start": 1.0, "end": 1.4},
                        {"word": "World", "start": 1.5, "end": 1.9},
                    ],
                },
            ]
        },
    ]
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "Hello", "start": 1.0, "end": 1.4},
            {"text": "World", "start": 1.5, "end": 1.9},
        ]
    }
    fake_wx.load_model.return_value = fake_model

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        result = align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            margin=0.1,
        )

    assert result == out
    fake_wx.load_model.assert_called_once()
    assert fake_wx.align.call_count == 2
    script_align_segments = fake_wx.align.call_args_list[1][0][0]
    # Short lines are merged for alignment; output cues stay separate.
    assert [seg["text"] for seg in script_align_segments] == ["Hello World"]
    cues = srt.load(out)
    assert [c.text for c in cues] == ["Hello", "World"]


def test_align_txt_uses_script_text_when_asr_differs(tmp_path: Path):
    """ASR wording must not be what WhisperX aligns; script cues are."""
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.txt"
    subtitle.write_text("Alright, that's it.\n", encoding="utf-8")
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 40, dtype=np.float32)
    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.side_effect = [
        {
            "segments": [
                {
                    "text": "All right that's it",
                    "start": 34.0,
                    "end": 36.0,
                    "words": [
                        {"word": "All", "start": 34.0, "end": 34.2},
                        {"word": "right", "start": 34.3, "end": 34.6},
                        {"word": "that's", "start": 35.0, "end": 35.3},
                        {"word": "it", "start": 35.4, "end": 35.8},
                    ],
                }
            ]
        },
        {
            "segments": [
                {
                    "text": "Alright, that's it.",
                    "start": 34.5,
                    "end": 35.8,
                    "words": [
                        {"word": "Alright,", "start": 34.5, "end": 34.9},
                        {"word": "that's", "start": 35.0, "end": 35.3},
                        {"word": "it.", "start": 35.4, "end": 35.8},
                    ],
                }
            ]
        },
    ]
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "All right that's it", "start": 34.0, "end": 36.0},
        ]
    }
    fake_wx.load_model.return_value = fake_model

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            margin=0.5,
        )

    assert fake_wx.align.call_count == 2
    script_align_segments = fake_wx.align.call_args_list[1][0][0]
    assert len(script_align_segments) == 1
    assert script_align_segments[0]["text"] == "Alright, that's it."
    assert script_align_segments[0]["start"] <= 34.0
    assert script_align_segments[0]["end"] >= 36.0
    cues = srt.load(out)
    assert cues[0].text == "Alright, that's it."
    assert abs(cues[0].end - 35.8) < 1e-6


def test_build_script_align_segments_uses_word_times_not_even_span():
    """Long ASR blobs must not evenly stretch cues across trailing silence."""
    from sub_align.align import build_script_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "The car's packed.", 0.0, 0.0),
        Cue(2, "Did you get the camera?", 0.0, 0.0),
    ]
    # Mega-segment end includes trailing silence; word times stay on speech.
    asr = [
        {
            "text": "The car's packed. Did you get the camera?",
            "start": 8.0,
            "end": 20.0,
            "words": [
                {"word": "The", "start": 8.9, "end": 9.0},
                {"word": "car's", "start": 9.0, "end": 9.3},
                {"word": "packed.", "start": 9.3, "end": 11.2},
                {"word": "Did", "start": 11.7, "end": 11.9},
                {"word": "you", "start": 11.9, "end": 12.1},
                {"word": "get", "start": 12.1, "end": 12.3},
                {"word": "the", "start": 12.3, "end": 12.4},
                {"word": "camera?", "start": 12.4, "end": 13.0},
            ],
        }
    ]
    segments = build_script_align_segments(cues, asr, margin=0.0, audio_duration=25.0)
    assert segments[1]["start"] == pytest.approx(11.7, abs=0.05)
    assert segments[1]["end"] == pytest.approx(13.0, abs=0.05)
    # Even interpolation across 8–20s pushes cue 2 several seconds late.
    asr_no_words = [
        {
            "text": "The car's packed. Did you get the camera?",
            "start": 8.0,
            "end": 20.0,
        }
    ]
    even = build_script_align_segments(cues, asr_no_words, margin=0.0, audio_duration=25.0)
    assert even[1]["start"] > segments[1]["start"] + 1.0


def test_segments_with_word_times_prefers_aligned_segments():
    from sub_align.align import _segments_with_word_times

    asr = [{"text": "Hello world", "start": 0.0, "end": 10.0}]
    aligned = {
        "segments": [
            {
                "text": "Hello world",
                "start": 1.0,
                "end": 2.0,
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 1.4},
                    {"word": "world", "start": 1.5, "end": 2.0},
                ],
            }
        ]
    }
    out = _segments_with_word_times(asr, aligned)
    assert out[0]["words"][0]["start"] == 1.0


def test_segments_with_word_times_falls_back_to_word_segments():
    from sub_align.align import _segments_with_word_times

    asr = [{"text": "Hello", "start": 0.0, "end": 5.0}]
    aligned = {
        "segments": [{"text": "Hello", "start": 1.0, "end": 2.0}],
        "word_segments": [{"word": "Hello", "start": 1.0, "end": 1.5}],
    }
    out = _segments_with_word_times(asr, aligned)
    assert len(out) == 1
    assert out[0]["words"][0]["end"] == 1.5


def test_build_script_align_segments_maps_alright_vs_all_right():
    from sub_align.align import build_script_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "So sue me.", 0.0, 0.0),
        Cue(2, "Alright, that's it.", 0.0, 0.0),
    ]
    asr = [
        {"text": "So sue me. All right that's it.", "start": 33.0, "end": 36.0},
    ]
    segments = build_script_align_segments(cues, asr, margin=0.0, audio_duration=40.0)
    assert [s["text"] for s in segments] == ["So sue me.", "Alright, that's it."]
    # Second cue should cover the latter part of the ASR span, including "it".
    assert segments[1]["start"] >= segments[0]["start"]
    assert segments[1]["end"] > segments[1]["start"]
    assert segments[1]["end"] >= 35.0


def test_build_script_align_segments_enforces_min_search_duration():
    from sub_align.align import build_script_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "Yep.", 0.0, 0.0),
        Cue(2, "Did you fill up the tank?", 0.0, 0.0),
    ]
    asr = [
        {"text": "Yep.", "start": 1.0, "end": 1.05},
        {"text": "Did you fill up the tank?", "start": 1.05, "end": 3.0},
    ]
    segments = build_script_align_segments(
        cues, asr, margin=0.0, audio_duration=10.0, min_search_duration=0.5
    )
    assert segments[0]["end"] - segments[0]["start"] >= 0.5 - 1e-9


def test_build_script_align_segments_snaps_missed_cue_to_vad_island():
    """ASR-skipped middle line must not search the whole gap into the next line."""
    from sub_align.align import build_script_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "Oh great, the stupid computer froze again.", 0.0, 0.0),
        Cue(2, "It's the third time today.", 0.0, 0.0),
        Cue(3, "Hey Samuel, can you come take a look at my PC?", 0.0, 0.0),
    ]
    # ASR skips cue 2; fill would otherwise span ~7.5 → 17.4.
    asr = [
        {
            "text": "Oh great, the stupid computer froze again.",
            "start": 5.4,
            "end": 7.5,
            "words": [
                {"word": "Oh", "start": 5.4, "end": 5.5},
                {"word": "great", "start": 5.5, "end": 5.8},
                {"word": "the", "start": 5.8, "end": 5.9},
                {"word": "stupid", "start": 5.9, "end": 6.3},
                {"word": "computer", "start": 6.3, "end": 6.8},
                {"word": "froze", "start": 6.8, "end": 7.2},
                {"word": "again.", "start": 7.2, "end": 7.5},
            ],
        },
        {
            "text": "Hey Samuel, can you come take a look at my PC?",
            "start": 16.5,
            "end": 18.3,
            "words": [
                {"word": "Hey", "start": 16.5, "end": 16.7},
                {"word": "Samuel", "start": 16.7, "end": 17.1},
                {"word": "can", "start": 17.1, "end": 17.2},
                {"word": "you", "start": 17.2, "end": 17.3},
                {"word": "come", "start": 17.3, "end": 17.5},
                {"word": "take", "start": 17.5, "end": 17.7},
                {"word": "a", "start": 17.7, "end": 17.8},
                {"word": "look", "start": 17.8, "end": 18.0},
                {"word": "at", "start": 18.0, "end": 18.1},
                {"word": "my", "start": 18.1, "end": 18.2},
                {"word": "PC?", "start": 18.2, "end": 18.3},
            ],
        },
    ]
    speech_spans = [(5.3, 7.6), (8.0, 10.0), (16.4, 18.4)]
    without = build_script_align_segments(cues, asr, margin=0.0, audio_duration=30.0)
    assert without[1]["end"] - without[1]["start"] > 8.0

    with_vad = build_script_align_segments(
        cues,
        asr,
        margin=0.0,
        audio_duration=30.0,
        speech_spans=speech_spans,
    )
    assert with_vad[1]["start"] == pytest.approx(8.0, abs=0.05)
    assert with_vad[1]["end"] == pytest.approx(10.0, abs=0.05)
    # Must not sit on the following utterance.
    assert with_vad[1]["end"] < 16.0


def test_build_script_align_segments_maps_unmatched_run_across_vad():
    from sub_align.align import build_script_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello there friend.", 0.0, 0.0),
        Cue(2, "Short one.", 0.0, 0.0),
        Cue(3, "Another line here.", 0.0, 0.0),
        Cue(4, "Finally done speaking.", 0.0, 0.0),
    ]
    asr = [
        {
            "text": "Hello there friend.",
            "start": 1.0,
            "end": 2.0,
            "words": [
                {"word": "Hello", "start": 1.0, "end": 1.3},
                {"word": "there", "start": 1.3, "end": 1.6},
                {"word": "friend.", "start": 1.6, "end": 2.0},
            ],
        },
        {
            "text": "Finally done speaking.",
            "start": 12.0,
            "end": 14.0,
            "words": [
                {"word": "Finally", "start": 12.0, "end": 12.5},
                {"word": "done", "start": 12.5, "end": 13.0},
                {"word": "speaking.", "start": 13.0, "end": 14.0},
            ],
        },
    ]
    speech_spans = [(1.0, 2.0), (4.0, 5.0), (6.0, 8.0), (12.0, 14.0)]
    segments = build_script_align_segments(
        cues,
        asr,
        margin=0.0,
        audio_duration=20.0,
        speech_spans=speech_spans,
    )
    assert segments[1]["start"] >= 4.0
    assert segments[2]["end"] <= 8.5
    assert segments[1]["end"] <= segments[2]["start"] + 1e-9
    assert segments[2]["end"] < 11.0
    assert segments[3]["start"] >= 11.5


def test_group_short_cues_attaches_to_neighbors():
    from sub_align.align import _group_short_cues, _merge_align_segments
    from sub_align.models import Cue

    cues = [
        Cue(1, "So are we all ready to go?", 0.0, 0.0),
        Cue(2, "Yep.", 0.0, 0.0),
        Cue(3, "I think so.", 0.0, 0.0),
        Cue(4, "The car's packed.", 0.0, 0.0),
    ]
    groups = _group_short_cues(cues)
    assert groups[0] == [0, 1, 2]
    assert groups[1] == [3]

    segments = [
        {"text": c.text, "start": float(i), "end": float(i) + 0.2} for i, c in enumerate(cues)
    ]
    merged, groups2 = _merge_align_segments(cues, segments)
    assert groups2 == groups
    assert len(merged) == 2
    assert merged[0]["text"] == "So are we all ready to go? Yep. I think so."


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


def test_apply_aligned_times_repairs_thin_mismatch():
    from sub_align.align import _apply_aligned_times
    from sub_align.models import Cue

    cues = [
        Cue(1, "Got it.", 0.0, 0.0),
        Cue(2, "Did you fill up the tank?", 0.0, 0.0),
    ]
    aligned = [
        {
            "text": "Got it. Did you fill up the tank?",
            "start": 1.0,
            "end": 3.0,
            "words": [
                # Token drift → count fallback; tiny span triggers repair.
                {"word": "Nope", "start": 1.70, "end": 1.71},
                {"word": "x", "start": 1.71, "end": 1.72},
                {"word": "Did", "start": 1.05, "end": 1.2},
                {"word": "you", "start": 1.2, "end": 1.3},
                {"word": "fill", "start": 1.3, "end": 1.5},
                {"word": "up", "start": 1.5, "end": 1.6},
                {"word": "the", "start": 1.6, "end": 1.7},
                {"word": "tank", "start": 1.8, "end": 2.5},
                {"word": "question", "start": 2.5, "end": 3.0},
            ],
        }
    ]
    search = [
        {"text": "Got it.", "start": 1.0, "end": 1.6},
        {"text": "Did you fill up the tank?", "start": 1.6, "end": 3.0},
    ]
    updated = _apply_aligned_times(
        cues,
        aligned,
        search_segments=search,
        groups=[[0, 1]],
    )
    # Count-fallback on "Got it" yields a tiny span → repaired from search window.
    assert abs(updated[0].start - 1.0) < 1e-6
    assert abs(updated[0].end - 1.6) < 1e-6


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
    assert updated[2] is cues[2] or (
        abs(updated[2].start - 18.0) < 1e-6 and abs(updated[2].end - 19.0) < 1e-6
    )


def test_trim_overlaps_advances_later_start_when_inverted():
    from sub_align.align import _trim_overlaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "A", 5.0, 6.0),
        Cue(2, "B", 4.0, 7.0),
    ]
    updated = _trim_overlaps(cues)
    assert abs(updated[0].start - 5.0) < 1e-6
    assert abs(updated[0].end - 6.0) < 1e-6
    assert abs(updated[1].start - 6.0) < 1e-6
    assert abs(updated[1].end - 7.0) < 1e-6
    assert updated[1].start >= updated[0].end


def test_trim_overlaps_zero_duration_inversion():
    from sub_align.align import _trim_overlaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "Got it.", 17.177, 17.177),
        Cue(2, "Did you fill up the tank?", 17.047, 19.652),
    ]
    updated = _trim_overlaps(cues)
    assert abs(updated[0].start - 17.177) < 1e-6
    assert abs(updated[1].start - 17.177) < 1e-6
    assert updated[1].start >= updated[0].end
    assert abs(updated[1].end - 19.652) < 1e-6


def test_fill_gaps_extends_ends_to_next_start():
    from sub_align.align import _fill_gaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "A", 1.0, 1.4),
        Cue(2, "B", 1.5, 1.9),
        Cue(3, "C", 2.5, 2.8),
    ]
    updated = _fill_gaps(cues, audio_duration=5.0)
    assert abs(updated[0].start - 1.0) < 1e-6
    assert abs(updated[0].end - 1.5) < 1e-6
    assert abs(updated[1].start - 1.5) < 1e-6
    assert abs(updated[1].end - 2.5) < 1e-6
    assert abs(updated[2].start - 2.5) < 1e-6
    assert abs(updated[2].end - 5.0) < 1e-6


def test_fill_gaps_keeps_last_end_without_duration():
    from sub_align.align import _fill_gaps
    from sub_align.models import Cue

    cues = [
        Cue(1, "A", 1.0, 1.4),
        Cue(2, "B", 1.5, 1.9),
    ]
    updated = _fill_gaps(cues, audio_duration=None)
    assert abs(updated[0].end - 1.5) < 1e-6
    assert abs(updated[1].end - 1.9) < 1e-6


def test_align_file_fill_gaps(tmp_path: Path):
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

        align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            margin=0.1,
            fill_gaps=True,
        )

    cues = srt.load(out)
    assert abs(cues[0].end - 1.5) < 1e-6
    assert abs(cues[1].end - 3.0) < 1e-6
