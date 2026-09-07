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
            {
                "text": "Hello World",
                "start": 1.0,
                "end": 1.9,
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 1.4},
                    {"word": "World", "start": 1.5, "end": 1.9},
                ],
            }
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
            auto_offset=False,
        )

    assert result == out
    assert fake_wx.align.called
    # Short cues are merged for FA.
    assert len(fake_wx.align.call_args.args[0]) == 1
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


def test_align_file_audio_only_transcribes(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 3, dtype=np.float32)
    fake_wx = MagicMock()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "Hello", "start": 1.0, "end": 1.4},
            {"text": "World", "start": 1.5, "end": 1.9},
        ]
    }
    fake_wx.load_model.return_value = fake_model
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {
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
    }

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        result = align_file(
            media=media,
            subtitle=None,
            output=out,
            language="en",
            device="cpu",
        )

    assert result == out
    fake_wx.load_model.assert_called_once()
    fake_wx.load_align_model.assert_called_once()
    fake_wx.align.assert_called_once()
    cues = srt.load(out)
    assert [c.text for c in cues] == ["Hello", "World"]
    assert abs(cues[0].start - 1.0) < 1e-6
    assert abs(cues[1].end - 1.9) < 1e-6


def test_align_file_audio_only_splits_long_transcription(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    out = tmp_path / "out.srt"

    text = (
        "Hello everyone and welcome back to the show. "
        "Today we are going to talk about how to build a consistent learning habit "
        "without burning out after the first few days."
    )
    fake_audio = np.zeros(16_000 * 20, dtype=np.float32)
    fake_wx = MagicMock()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": [{"text": text, "start": 1.0, "end": 12.0}]}
    fake_wx.load_model.return_value = fake_model
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    # Even proportional fallback still works if align returns no words.
    fake_wx.align.return_value = {"segments": [{"text": text, "start": 1.0, "end": 12.0}]}

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        result = align_file(
            media=media,
            subtitle=None,
            output=out,
            language="en",
            device="cpu",
        )

    assert result == out
    cues = srt.load(out)
    # Default: strong punctuation only → two sentences, second stays whole.
    assert len(cues) == 2
    assert cues[0].text.rstrip().endswith(".")
    assert "learning habit" in cues[1].text
    assert cues[1].start >= cues[0].end
    assert abs(cues[-1].end - 12.0) < 1e-6


def test_align_file_audio_only_length_limits_split_further(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    out = tmp_path / "out.srt"

    text = (
        "Today we are going to talk about how to build a consistent learning habit "
        "without burning out after the first few days."
    )
    fake_audio = np.zeros(16_000 * 20, dtype=np.float32)
    fake_wx = MagicMock()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": [{"text": text, "start": 1.0, "end": 12.0}]}
    fake_wx.load_model.return_value = fake_model
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {"segments": [{"text": text, "start": 1.0, "end": 12.0}]}

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        result = align_file(
            media=media,
            subtitle=None,
            output=out,
            language="en",
            device="cpu",
            max_words=12,
            max_chars=42,
        )

    assert result == out
    cues = srt.load(out)
    assert len(cues) >= 2
    assert all(len(c.text.split()) <= 12 for c in cues)
    assert cues[1].start >= cues[0].end


def test_align_file_audio_only_keeps_short_cues_and_uses_word_times(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    out = tmp_path / "out.srt"

    text = (
        "Oh, I don't know if you've heard, but someone moved into that old house "
        "down the road. Yeah, I know."
    )
    words = [
        {"word": "Oh,", "start": 8.1, "end": 8.3},
        {"word": "I", "start": 8.3, "end": 8.4},
        {"word": "don't", "start": 8.4, "end": 8.6},
        {"word": "know", "start": 8.6, "end": 8.8},
        {"word": "if", "start": 8.8, "end": 8.9},
        {"word": "you've", "start": 8.9, "end": 9.1},
        {"word": "heard,", "start": 9.1, "end": 9.5},
        {"word": "but", "start": 9.6, "end": 9.8},
        {"word": "someone", "start": 9.8, "end": 10.2},
        {"word": "moved", "start": 10.2, "end": 10.5},
        {"word": "into", "start": 10.5, "end": 10.7},
        {"word": "that", "start": 10.7, "end": 10.9},
        {"word": "old", "start": 10.9, "end": 11.1},
        {"word": "house", "start": 11.1, "end": 11.4},
        {"word": "down", "start": 11.4, "end": 11.6},
        {"word": "the", "start": 11.6, "end": 11.7},
        {"word": "road.", "start": 11.7, "end": 12.1},
        {"word": "Yeah,", "start": 12.3, "end": 12.6},
        {"word": "I", "start": 12.6, "end": 12.7},
        {"word": "know.", "start": 12.7, "end": 13.2},
    ]
    fake_audio = np.zeros(16_000 * 20, dtype=np.float32)
    fake_wx = MagicMock()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": [{"text": text, "start": 8.1, "end": 13.2}]}
    fake_wx.load_model.return_value = fake_model
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {
        "segments": [{"text": text, "start": 8.1, "end": 13.2, "words": words}]
    }

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
    ):
        result = align_file(
            media=media,
            subtitle=None,
            output=out,
            language="en",
            device="cpu",
        )

    assert result == out
    cues = srt.load(out)
    texts = [c.text for c in cues]
    assert any(t == "Yeah, I know." or t.endswith("Yeah, I know.") for t in texts)
    # Short reply must stay its own cue, not merged into the previous sentence.
    yeah_i = next(i for i, t in enumerate(texts) if "Yeah, I know." in t)
    assert texts[yeah_i].strip() == "Yeah, I know."
    # Default keeps the full sentence through "road."
    first = cues[0]
    assert "road." in first.text
    assert "Yeah" not in first.text
    assert abs(first.end - 12.1) < 1e-6
    assert cues[yeah_i].start >= 12.0


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
            {
                "text": "Hello World",
                "start": 1.0,
                "end": 1.9,
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 1.4},
                    {"word": "World", "start": 1.5, "end": 1.9},
                ],
            }
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
            auto_offset=False,
        )

    cues = srt.load(out)
    assert abs(cues[0].end - 1.5) < 1e-6
    assert abs(cues[1].end - 3.0) < 1e-6


def test_align_file_trim_start_shifts_output_times(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.txt"
    subtitle.write_text("Hello\nWorld\n", encoding="utf-8")
    out = tmp_path / "out.srt"

    # 10s media; intro occupies first 2s (trimmed away).
    fake_audio = np.zeros(16_000 * 10, dtype=np.float32)
    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    # Times returned by WhisperX are relative to the trimmed audio.
    fake_wx.align.return_value = {
        "segments": [
            {"text": "Hello", "start": 0.5, "end": 1.0},
            {"text": "World", "start": 1.2, "end": 1.8},
        ],
        "word_segments": [
            {"word": "Hello", "start": 0.5, "end": 1.0},
            {"word": "World", "start": 1.2, "end": 1.8},
        ],
    }
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "Hello", "start": 0.5, "end": 1.0},
            {"text": "World", "start": 1.2, "end": 1.8},
        ]
    }
    fake_wx.load_model.return_value = fake_model

    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
        patch(
            "sub_align.align.detect_speech_spans",
            return_value=[(0.0, 8.0)],
        ),
    ):
        from sub_align.align import align_file

        align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            margin=0.0,
            trim_start=2.0,
            trim_end=0.0,
        )

    # Align must see 8s of audio (10 - 2).
    aligned_audio = fake_wx.align.call_args.args[3]
    assert abs(len(aligned_audio) / 16_000 - 8.0) < 1e-6

    cues = srt.load(out)
    assert abs(cues[0].start - 2.5) < 1e-6
    assert abs(cues[0].end - 3.0) < 1e-6
    assert abs(cues[1].start - 3.2) < 1e-6
    assert abs(cues[1].end - 3.8) < 1e-6


def test_estimate_global_offset_median_shift():
    from sub_align.align import estimate_global_offset
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello there friend", 0.0, 1.0),
        Cue(2, "How are you today", 1.0, 2.0),
        Cue(3, "I am doing fine", 2.0, 3.0),
        Cue(4, "Thanks for asking me", 3.0, 4.0),
    ]
    # ASR is consistently ~10s later than cue times.
    asr = [
        {"text": "Hello there friend", "start": 10.0, "end": 11.0},
        {"text": "How are you today", "start": 11.0, "end": 12.0},
        {"text": "I am doing fine", "start": 12.0, "end": 13.0},
        {"text": "Thanks for asking me", "start": 13.0, "end": 14.0},
    ]
    assert abs(estimate_global_offset(cues, asr) - 10.0) < 0.5


def test_estimate_global_offset_ignores_small_drift():
    from sub_align.align import estimate_global_offset
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello there friend", 0.0, 1.0),
        Cue(2, "How are you today", 1.0, 2.0),
        Cue(3, "I am doing fine", 2.0, 3.0),
    ]
    asr = [
        {"text": "Hello there friend", "start": 0.1, "end": 1.1},
        {"text": "How are you today", "start": 1.1, "end": 2.1},
        {"text": "I am doing fine", "start": 2.1, "end": 3.1},
    ]
    assert estimate_global_offset(cues, asr) == 0.0


def test_estimate_global_offset_needs_min_matches():
    from sub_align.align import estimate_global_offset
    from sub_align.models import Cue

    cues = [
        Cue(1, "Hello there", 0.0, 1.0),
        Cue(2, "How are you", 1.0, 2.0),
    ]
    asr = [
        {"text": "Hello there", "start": 5.0, "end": 6.0},
        {"text": "How are you", "start": 6.0, "end": 7.0},
    ]
    assert estimate_global_offset(cues, asr) == 0.0


def test_align_file_manual_offset(tmp_path: Path):
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

    fake_audio = np.zeros(16_000 * 5, dtype=np.float32)
    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {
        "segments": [
            {
                "text": "Hello World",
                "start": 2.0,
                "end": 2.9,
                "words": [
                    {"word": "Hello", "start": 2.0, "end": 2.4},
                    {"word": "World", "start": 2.5, "end": 2.9},
                ],
            }
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
            offset=2.0,
            auto_offset=True,
        )

    # Manual offset must skip transcription.
    fake_wx.load_model.assert_not_called()
    # Short cues merge into one FA segment spanning the shifted windows.
    segments = fake_wx.align.call_args.args[0]
    assert len(segments) == 1
    assert segments[0]["start"] == pytest.approx(1.9, abs=1e-6)
    assert "Hello" in segments[0]["text"] and "World" in segments[0]["text"]

    cues = srt.load(out)
    assert abs(cues[0].start - 2.0) < 1e-6
    assert abs(cues[1].end - 2.9) < 1e-6


def test_align_file_auto_offset(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    # Cues are 10s early relative to speech in the media.
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:00,000 --> 00:00:01,000
Hello there friend

2
00:00:01,000 --> 00:00:02,000
How are you today

3
00:00:02,000 --> 00:00:03,000
I am doing fine

4
00:00:03,000 --> 00:00:04,000
Thanks for asking me
"""
        ),
    )
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 20, dtype=np.float32)
    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_wx.align.return_value = {
        "segments": [
            {"text": "Hello there friend", "start": 10.0, "end": 10.8},
            {"text": "How are you today", "start": 11.0, "end": 11.8},
            {"text": "I am doing fine", "start": 12.0, "end": 12.8},
            {"text": "Thanks for asking me", "start": 13.0, "end": 13.8},
        ]
    }
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "segments": [
            {"text": "Hello there friend", "start": 10.0, "end": 11.0},
            {"text": "How are you today", "start": 11.0, "end": 12.0},
            {"text": "I am doing fine", "start": 12.0, "end": 13.0},
            {"text": "Thanks for asking me", "start": 13.0, "end": 14.0},
        ]
    }
    fake_wx.load_model.return_value = fake_model

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
        )

    fake_wx.load_model.assert_called_once()
    segments = fake_wx.align.call_args.args[0]
    # After ~+10s auto offset, refine windows sit near the ASR times.
    assert segments[0]["start"] == pytest.approx(9.9, abs=0.6)
    cues = srt.load(out)
    assert abs(cues[0].start - 10.0) < 1e-6


def test_align_file_trim_rejects_over_trim(tmp_path: Path):
    from sub_align.align import align_file

    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.txt"
    subtitle.write_text("Hi\n", encoding="utf-8")
    fake_audio = np.zeros(16_000 * 2, dtype=np.float32)

    with (
        patch.dict(sys.modules, {"whisperx": MagicMock()}),
        patch("sub_align.align.load_audio", return_value=fake_audio),
        pytest.raises(ValueError, match="less than audio duration"),
    ):
        align_file(
            media=media,
            subtitle=subtitle,
            language="en",
            device="cpu",
            trim_start=1.5,
            trim_end=1.0,
        )


def test_dialogue_lines_detects_dash_multiline():
    from sub_align.align import _dialogue_lines

    text = "-Everything is going according to plan.\n-Until it isn't."
    lines = _dialogue_lines(text)
    assert lines is not None
    assert len(lines) == 2
    assert _dialogue_lines("-Only one speaker line") is None
    assert _dialogue_lines("No dashes\nJust two lines") is None
    assert _dialogue_lines("-A\n-B\n-C") is not None
    assert len(_dialogue_lines("-A\n-B\n-C") or []) == 3


def test_expand_dialogue_align_segments_splits_and_floors_tail():
    from sub_align.align import _MIN_SEARCH_DURATION, _expand_dialogue_align_segments
    from sub_align.models import Cue

    cue = Cue(
        1,
        "-Everything is going according to plan.\n-Until it isn't.",
        10.0,
        14.0,
    )
    segments = [{"text": cue.text, "start": 9.75, "end": 14.5}]
    expanded, ranges = _expand_dialogue_align_segments([cue], segments)
    assert ranges == [(0, 2)]
    assert len(expanded) == 2
    assert expanded[0]["text"].startswith("-Everything")
    assert expanded[1]["text"].startswith("-Until")
    assert expanded[0]["start"] == pytest.approx(9.75)
    assert expanded[1]["end"] == pytest.approx(14.5)
    assert expanded[1]["end"] - expanded[1]["start"] >= _MIN_SEARCH_DURATION - 1e-9
    assert expanded[0]["end"] <= expanded[1]["start"] + 1e-9


def test_clamp_refine_starts_restores_original_when_fa_enters_silence():
    from sub_align.align import _clamp_refine_starts
    from sub_align.models import Cue

    original = [Cue(1, "Hello there", 2.0, 3.5)]
    # FA pulled ~1s earlier into silence; speech starts at 2.0
    aligned = [Cue(1, "Hello there", 1.0, 3.5)]
    search = [{"text": "Hello there", "start": 0.75, "end": 4.0}]
    speech_spans = [(2.0, 3.6)]
    fixed = _clamp_refine_starts(
        original,
        aligned,
        search_segments=search,
        speech_spans=speech_spans,
    )
    assert fixed[0].start == pytest.approx(2.0)


def test_clamp_refine_starts_snaps_forward_to_speech():
    from sub_align.align import _clamp_refine_starts
    from sub_align.models import Cue

    original = [Cue(1, "Hi", 0.5, 2.0)]  # also in silence
    aligned = [Cue(1, "Hi", 0.5, 2.0)]
    search = [{"text": "Hi", "start": 0.25, "end": 2.5}]
    speech_spans = [(1.2, 2.0)]
    fixed = _clamp_refine_starts(
        original,
        aligned,
        search_segments=search,
        speech_spans=speech_spans,
    )
    assert fixed[0].start == pytest.approx(1.2)


def test_align_file_splits_dialogue_cue_for_force_align(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:10,000 --> 00:00:14,000
-Everything is going according to plan.
-Until it isn't.
"""
        ),
    )
    out = tmp_path / "out.srt"

    # Speech from 10s–14s; leading silence so VAD clamp has a clear island.
    fake_audio = np.zeros(16_000 * 16, dtype=np.float32)
    fake_audio[10 * 16_000 : 14 * 16_000] = 0.25

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def fake_align(segments, *args, **kwargs):
        # Expect two per-line segments for the dialogue cue.
        assert len(segments) == 2
        assert segments[0]["text"].startswith("-Everything")
        assert segments[1]["text"].startswith("-Until")
        return {
            "segments": [
                {
                    "text": segments[0]["text"],
                    "start": 10.0,
                    "end": 13.2,
                    "words": [
                        {"word": "Everything", "start": 10.0, "end": 10.4},
                        {"word": "is", "start": 10.4, "end": 10.5},
                        {"word": "going", "start": 10.5, "end": 10.8},
                        {"word": "according", "start": 10.8, "end": 11.2},
                        {"word": "to", "start": 11.2, "end": 11.3},
                        {"word": "plan", "start": 11.3, "end": 13.2},
                    ],
                },
                {
                    "text": segments[1]["text"],
                    "start": 13.2,
                    "end": 14.0,
                    "words": [
                        {"word": "Until", "start": 13.2, "end": 13.5},
                        {"word": "it", "start": 13.5, "end": 13.6},
                        {"word": "isn't", "start": 13.6, "end": 14.0},
                    ],
                },
            ]
        }

    fake_wx.align.side_effect = fake_align

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
            margin=0.5,
            auto_offset=False,
        )

    assert result == out
    cues = srt.load(out)
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(10.0, abs=0.05)
    assert cues[0].end == pytest.approx(14.0, abs=0.05)
    assert "Until it isn't" in cues[0].text


def test_align_file_clamps_early_start_into_silence(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:02,000 --> 00:00:03,500
Hello there
"""
        ),
    )
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 5, dtype=np.float32)
    fake_audio[2 * 16_000 : int(3.5 * 16_000)] = 0.3

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    # FA incorrectly places start ~1s earlier into silence.
    fake_wx.align.return_value = {
        "segments": [
            {
                "text": "Hello there",
                "start": 1.0,
                "end": 3.5,
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 1.4},
                    {"word": "there", "start": 1.4, "end": 3.5},
                ],
            }
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
            margin=0.5,
            auto_offset=False,
        )

    cues = srt.load(out)
    assert cues[0].start == pytest.approx(2.0, abs=0.15)


def test_prefer_original_short_cues_keeps_collapsed_fa():
    from sub_align.align import _prefer_original_short_cues
    from sub_align.models import Cue

    original = [
        Cue(1, "Yes.", 11.464, 11.950),
        Cue(2, "Longer line that should keep FA times", 12.0, 14.0),
    ]
    aligned = [
        Cue(1, "Yes.", 12.100, 12.100),
        Cue(2, "Longer line that should keep FA times", 12.2, 13.8),
    ]
    fixed = _prefer_original_short_cues(original, aligned)
    assert fixed[0].start == pytest.approx(11.464)
    assert fixed[0].end == pytest.approx(11.950)
    assert fixed[1].start == pytest.approx(12.2)
    assert fixed[1].end == pytest.approx(13.8)


def test_prefer_original_short_cues_keeps_ballooned_window():
    from sub_align.align import _prefer_original_short_cues
    from sub_align.models import Cue

    original = [Cue(1, "Yes.", 11.464, 11.950)]
    # Thin-repair fallback to a wide search window.
    aligned = [Cue(1, "Yes.", 11.214, 12.450)]
    fixed = _prefer_original_short_cues(original, aligned)
    assert fixed[0].start == pytest.approx(11.464)
    assert fixed[0].end == pytest.approx(11.950)


def test_prefer_original_short_cues_ignores_untimed():
    from sub_align.align import _prefer_original_short_cues
    from sub_align.models import Cue

    original = [Cue(1, "Yes.", 0.0, 0.0)]
    aligned = [Cue(1, "Yes.", 1.0, 1.2)]
    fixed = _prefer_original_short_cues(original, aligned)
    assert fixed[0].start == pytest.approx(1.0)
    assert fixed[0].end == pytest.approx(1.2)


def test_repair_collapsed_cues_restores_original():
    from sub_align.align import _repair_collapsed_cues
    from sub_align.models import Cue

    original = [Cue(1, "No.", 12.020, 12.427)]
    aligned = [Cue(1, "No.", 12.100, 12.100)]
    fixed = _repair_collapsed_cues(original, aligned)
    assert fixed[0].start == pytest.approx(12.020)
    assert fixed[0].end == pytest.approx(12.427)


def test_align_file_srt_merges_short_cues_for_force_align(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:09,217 --> 00:00:11,203
Really? Would it be totally weird if I used it?

2
00:00:11,464 --> 00:00:11,950
Yes.

3
00:00:12,020 --> 00:00:12,427
No.
"""
        ),
    )
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 16, dtype=np.float32)
    fake_audio[9 * 16_000 : 13 * 16_000] = 0.25

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def fake_align(segments, *args, **kwargs):
        # Short Yes/No attach to the previous long cue for FA.
        assert len(segments) == 1
        assert "Yes." in segments[0]["text"]
        assert "No." in segments[0]["text"]
        return {
            "segments": [
                {
                    "text": segments[0]["text"],
                    "start": 9.5,
                    "end": 12.4,
                    "words": [
                        {"word": "Really", "start": 9.51, "end": 9.7},
                        {"word": "Would", "start": 9.8, "end": 10.0},
                        {"word": "it", "start": 10.0, "end": 10.1},
                        {"word": "be", "start": 10.1, "end": 10.2},
                        {"word": "totally", "start": 10.2, "end": 10.5},
                        {"word": "weird", "start": 10.5, "end": 10.8},
                        {"word": "if", "start": 10.8, "end": 10.9},
                        {"word": "I", "start": 10.9, "end": 11.0},
                        {"word": "used", "start": 11.0, "end": 11.1},
                        {"word": "it", "start": 11.1, "end": 11.16},
                        {"word": "Yes", "start": 11.46, "end": 11.9},
                        {"word": "No", "start": 12.02, "end": 12.4},
                    ],
                }
            ]
        }

    fake_wx.align.side_effect = fake_align

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
            margin=0.5,
            auto_offset=False,
        )

    assert result == out
    cues = srt.load(out)
    assert len(cues) == 3
    assert cues[1].text == "Yes."
    assert cues[1].start == pytest.approx(11.46, abs=0.05)
    assert cues[1].end == pytest.approx(11.9, abs=0.05)
    assert cues[1].end - cues[1].start >= 0.05
    assert cues[2].start == pytest.approx(12.02, abs=0.05)


def test_align_file_srt_restores_collapsed_short_cue(tmp_path: Path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    subtitle = tmp_path / "a.srt"
    srt.dump(
        subtitle,
        srt.loads(
            """1
00:00:11,464 --> 00:00:11,950
Yes.

2
00:00:12,020 --> 00:00:12,427
No.
"""
        ),
    )
    out = tmp_path / "out.srt"

    fake_audio = np.zeros(16_000 * 16, dtype=np.float32)
    fake_audio[11 * 16_000 : 13 * 16_000] = 0.25

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})

    def fake_align(segments, *args, **kwargs):
        # Merged short cues, but FA returns a collapsed Yes timestamp.
        assert len(segments) == 1
        return {
            "segments": [
                {
                    "text": segments[0]["text"],
                    "start": 12.1,
                    "end": 12.3,
                    "words": [
                        {"word": "Yes", "start": 12.1, "end": 12.1},
                        {"word": "No", "start": 12.115, "end": 12.3},
                    ],
                }
            ]
        }

    fake_wx.align.side_effect = fake_align

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
            margin=0.5,
            auto_offset=False,
        )

    cues = srt.load(out)
    assert cues[0].text == "Yes."
    assert cues[0].end - cues[0].start >= 0.05
    assert cues[0].start == pytest.approx(11.464, abs=0.05)
    assert cues[0].end == pytest.approx(11.950, abs=0.05)
