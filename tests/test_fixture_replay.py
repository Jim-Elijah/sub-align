"""Replay recorded WhisperX JSON fixtures through ``align_file`` (no live model).

Uses ``tests/fixtures/clip*_timed.wav`` + ``*.whisperx.json`` recorded by
``scripts/record_whisperx_fixtures.py``. ASR ``transcribe`` / first ``align``
are replayed from JSON; script force-align maps cue tokens onto the recorded
word timeline (fallback: even split inside the search window).
"""

from __future__ import annotations

import json
import re
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sub_align.formats import srt
from sub_align.vad import SAMPLE_RATE

FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ALNUM = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, path
        assert wf.getsampwidth() == 2, path
        assert wf.getframerate() == SAMPLE_RATE, path
        raw = wf.readframes(wf.getnframes())
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0).copy()


def _load_payload(stem: str) -> dict[str, Any]:
    path = FIXTURES / f"{stem}.whisperx.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_seg_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _alnum_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _ALNUM.finditer(text)]


def _looks_like_asr(segments: list[dict], asr_segments: list[dict]) -> bool:
    if len(segments) != len(asr_segments):
        return False
    for left, right in zip(segments, asr_segments, strict=True):
        if _norm_seg_text(left.get("text", "")) != _norm_seg_text(right.get("text", "")):
            return False
    return True


def _word_bank(asr_aligned: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Flatten aligned words to (alnum_token, start, end)."""
    bank: list[tuple[str, float, float]] = []
    words = list(asr_aligned.get("word_segments") or [])
    if not words:
        for seg in asr_aligned.get("segments") or []:
            words.extend(seg.get("words") or [])
    for word in words:
        raw = str(word.get("word") or "")
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            continue
        tokens = _alnum_tokens(raw)
        if not tokens:
            continue
        # Multi-token ASR words (rare): share the same span.
        for token in tokens:
            bank.append((token, float(start), float(end)))
    return bank


def _even_words(text: str, start: float, end: float) -> list[dict[str, Any]]:
    tokens = text.split()
    if not tokens:
        return []
    dur = max(end - start, 0.05)
    step = dur / len(tokens)
    return [
        {
            "word": token,
            "start": start + i * step,
            "end": start + (i + 1) * step,
            "score": 0.9,
        }
        for i, token in enumerate(tokens)
    ]


def _map_segment_words(
    text: str,
    *,
    window_start: float,
    window_end: float,
    bank: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    """Attach recorded word times to script tokens via SequenceMatcher."""
    display_tokens = text.split()
    if not display_tokens:
        return []
    cue_toks = _alnum_tokens(text)
    if not cue_toks or not bank:
        return _even_words(text, window_start, window_end)

    bank_toks = [t for t, _, _ in bank]
    matcher = SequenceMatcher(a=cue_toks, b=bank_toks, autojunk=False)
    cue_times: list[tuple[float, float] | None] = [None] * len(cue_toks)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            _tok, t0, t1 = bank[j1 + offset]
            cue_times[i1 + offset] = (t0, t1)

    # Map alnum cue tokens back onto whitespace display tokens.
    words: list[dict[str, Any]] = []
    cue_i = 0
    for token in display_tokens:
        alnum = _alnum_tokens(token)
        if not alnum:
            continue
        spans = []
        for _ in alnum:
            if cue_i < len(cue_times) and cue_times[cue_i] is not None:
                spans.append(cue_times[cue_i])
            cue_i += 1
        if spans:
            w0 = min(s[0] for s in spans if s is not None)
            w1 = max(s[1] for s in spans if s is not None)
        else:
            # Leave a hole; filled below from neighbors / window.
            words.append({"word": token, "start": None, "end": None, "score": 0.5})
            continue
        words.append({"word": token, "start": w0, "end": w1, "score": 0.95})

    # Fill unmatched display tokens from neighbors inside the search window.
    for i, word in enumerate(words):
        if word["start"] is not None:
            continue
        prev_end = window_start
        for j in range(i - 1, -1, -1):
            if words[j]["end"] is not None:
                prev_end = float(words[j]["end"])
                break
        next_start = window_end
        for j in range(i + 1, len(words)):
            if words[j]["start"] is not None:
                next_start = float(words[j]["start"])
                break
        mid0 = prev_end
        mid1 = max(prev_end + 0.05, min(next_start, prev_end + 0.2))
        word["start"] = mid0
        word["end"] = mid1
        word["score"] = 0.5

    if not words or any(w["start"] is None for w in words):
        return _even_words(text, window_start, window_end)
    return words  # type: ignore[return-value]


def _script_align(segments: list[dict], bank: list[tuple[str, float, float]]) -> dict[str, Any]:
    out_segments: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text") or "")
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        words = _map_segment_words(text, window_start=start, window_end=end, bank=bank)
        if words:
            start = float(words[0]["start"])
            end = float(words[-1]["end"])
        out_segments.append({"text": text, "start": start, "end": end, "words": words})
        flat_words.extend(words)
    return {"segments": out_segments, "word_segments": flat_words}


def _make_whisperx(payload: dict[str, Any]) -> MagicMock:
    asr = payload["asr"]
    asr_aligned = payload["asr_aligned"]
    asr_segments = list(asr.get("segments") or [])
    bank = _word_bank(asr_aligned)

    fake_wx = MagicMock()
    fake_wx.load_align_model.return_value = (MagicMock(), {"language": "en"})
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "language": asr.get("language", "en"),
        "segments": asr_segments,
    }
    fake_wx.load_model.return_value = fake_model

    def fake_align(segments, *args, **kwargs):
        segs = list(segments or [])
        if _looks_like_asr(segs, asr_segments):
            return {
                "segments": list(asr_aligned.get("segments") or []),
                "word_segments": list(asr_aligned.get("word_segments") or []),
            }
        return _script_align(segs, bank)

    fake_wx.align.side_effect = fake_align
    return fake_wx


def _run_align_file(
    *,
    media: Path,
    subtitle: Path | None,
    audio: np.ndarray,
    payload: dict,
    **kwargs,
):
    fake_wx = _make_whisperx(payload)
    out = kwargs.pop("output")
    with (
        patch.dict(sys.modules, {"whisperx": fake_wx}),
        patch("sub_align.align.load_audio", return_value=audio),
    ):
        from sub_align.align import align_file

        result = align_file(
            media=media,
            subtitle=subtitle,
            output=out,
            language="en",
            device="cpu",
            **kwargs,
        )
    return result, fake_wx


@pytest.fixture(scope="module")
def clip1_audio() -> np.ndarray:
    return _load_wav(FIXTURES / "clip1_timed.wav")


@pytest.fixture(scope="module")
def clip1_payload() -> dict[str, Any]:
    return _load_payload("clip1_timed")


@pytest.fixture(scope="module")
def clip2_audio() -> np.ndarray:
    return _load_wav(FIXTURES / "clip2_timed.wav")


@pytest.fixture(scope="module")
def clip2_payload() -> dict[str, Any]:
    return _load_payload("clip2_timed")


def test_fixture_path_a_audio_only(tmp_path: Path, clip1_audio: np.ndarray, clip1_payload: dict):
    media = FIXTURES / "clip1_timed.wav"
    out = tmp_path / "clip1.asr.srt"
    result, fake_wx = _run_align_file(
        media=media,
        subtitle=None,
        audio=clip1_audio,
        payload=clip1_payload,
        output=out,
    )
    assert result == out
    assert fake_wx.align.call_count == 1
    cues = srt.load(out)
    texts = [c.text.strip() for c in cues]
    assert any("Hello" in t for t in texts)
    assert any("fox" in t.lower() for t in texts)
    assert cues[0].start == pytest.approx(1.725, abs=0.15)
    assert cues[-1].end == pytest.approx(39.164, abs=0.5)
    assert all(c.end > c.start for c in cues)


def test_fixture_path_b_txt_keeps_script(
    tmp_path: Path, clip1_audio: np.ndarray, clip1_payload: dict
):
    media = FIXTURES / "clip1_timed.wav"
    subtitle = FIXTURES / "clip1_script.txt"
    out = tmp_path / "clip1_script.aligned.srt"
    result, fake_wx = _run_align_file(
        media=media,
        subtitle=subtitle,
        audio=clip1_audio,
        payload=clip1_payload,
        output=out,
        margin=0.5,
    )
    assert result == out
    assert fake_wx.align.call_count == 2
    cues = srt.load(out)
    texts = [c.text for c in cues]
    assert "All right, that is it." in texts
    assert "Yes" in texts
    assert texts.count("One") == 1
    assert texts.count("Two") == 1
    assert texts.count("Three") == 1
    # Script wording must win over ASR ("1 2 3" / "All right").
    assert not any(t.strip() in {"1", "2", "3"} for t in texts)
    hello = next(c for c in cues if c.text.strip() == "Hello")
    assert hello.start == pytest.approx(1.725, abs=0.4)
    assert all(c.end >= c.start for c in cues)


def test_fixture_path_c_srt_refine(tmp_path: Path, clip1_audio: np.ndarray, clip1_payload: dict):
    media = FIXTURES / "clip1_timed.wav"
    subtitle = FIXTURES / "clip1_drift.srt"
    out = tmp_path / "clip1_drift.aligned.srt"
    result, fake_wx = _run_align_file(
        media=media,
        subtitle=subtitle,
        audio=clip1_audio,
        payload=clip1_payload,
        output=out,
        margin=0.5,
        auto_offset=True,
    )
    assert result == out
    assert fake_wx.load_model.called
    assert fake_wx.align.call_count == 1
    cues = srt.load(out)
    assert len(cues) == 11
    assert cues[0].text.strip() == "Hello."
    assert "Alright" in cues[1].text or "All right" in cues[1].text
    # Drifted cue started at 0.7s; refine/offset should land near speech (~1.7s).
    assert cues[0].start == pytest.approx(1.7, abs=0.8)
    assert cues[4].text.startswith("The quick brown fox")
    assert all(c.end >= c.start for c in cues)


def test_fixture_path_c_dialogue_splits(
    tmp_path: Path, clip2_audio: np.ndarray, clip2_payload: dict
):
    media = FIXTURES / "clip2_timed.wav"
    subtitle = FIXTURES / "clip2_dialogue.srt"
    out = tmp_path / "clip2_dialogue.aligned.srt"
    result, fake_wx = _run_align_file(
        media=media,
        subtitle=subtitle,
        audio=clip2_audio,
        payload=clip2_payload,
        output=out,
        margin=0.5,
        auto_offset=False,
    )
    assert result == out
    # Dialogue expand → multiple FA segments in one align call.
    fa_segments = fake_wx.align.call_args.args[0]
    assert len(fa_segments) >= 5
    cues = srt.load(out)
    assert len(cues) == 3
    assert "Where are you going?" in cues[0].text
    assert "I'm going home now." in cues[0].text
    assert cues[0].start == pytest.approx(1.0, abs=0.6)
    assert cues[2].end == pytest.approx(18.0, abs=1.0)
    assert all(c.end > c.start for c in cues)
