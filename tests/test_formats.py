from __future__ import annotations

from sub_align.formats import lrc, srt
from sub_align.models import Cue


def test_srt_roundtrip():
    original = """1
00:00:01,000 --> 00:00:02,500
Hello world

2
00:00:03,000 --> 00:00:04,000
Second line
"""
    cues = srt.loads(original)
    assert len(cues) == 2
    assert cues[0].text == "Hello world"
    assert abs(cues[0].start - 1.0) < 1e-6
    assert abs(cues[0].end - 2.5) < 1e-6
    dumped = srt.dumps(cues)
    again = srt.loads(dumped)
    assert again[0].text == cues[0].text
    assert abs(again[1].start - 3.0) < 1e-6


def test_srt_wide_in_narrow_out():
    original = """1
0:0:1.5 --> 00:00:02.50
Hello

2
00:00:03,05 --> 00:00:04.500
World
"""
    cues = srt.loads(original)
    assert abs(cues[0].start - 1.5) < 1e-6
    assert abs(cues[0].end - 2.5) < 1e-6
    assert abs(cues[1].start - 3.05) < 1e-6
    assert abs(cues[1].end - 4.5) < 1e-6
    dumped = srt.dumps(cues)
    assert "00:00:01,500 --> 00:00:02,500" in dumped
    assert "00:00:03,050 --> 00:00:04,500" in dumped


def test_lrc_roundtrip_and_end_from_next():
    original = """[00:01.00]First
[00:03.50]Second
[00:05.00]Third
"""
    cues = lrc.loads(original)
    assert len(cues) == 3
    assert cues[0].text == "First"
    assert abs(cues[0].start - 1.0) < 1e-6
    assert abs(cues[0].end - 3.5) < 1e-6
    assert abs(cues[2].end - (5.0 + 3.0)) < 1e-6
    dumped = lrc.dumps(cues)
    again = lrc.loads(dumped)
    assert [c.text for c in again] == [c.text for c in cues]


def test_lrc_last_end_clamped_to_audio_duration():
    original = """[00:01.00]First
[00:05.00]Last
"""
    cues = lrc.loads(original, audio_duration=6.5)
    assert abs(cues[1].end - 6.5) < 1e-6
    cues_short = lrc.loads(original, audio_duration=10.0)
    assert abs(cues_short[1].end - 8.0) < 1e-6


def test_lrc_wide_in_narrow_out():
    original = """[0:01.5]First
[00:03,50]Second
[1:05.500]Third
[02:00.05]Fourth
"""
    cues = lrc.loads(original)
    assert abs(cues[0].start - 1.5) < 1e-6
    assert abs(cues[1].start - 3.5) < 1e-6
    assert abs(cues[2].start - 65.5) < 1e-6
    assert abs(cues[3].start - 120.05) < 1e-6
    dumped = lrc.dumps(cues)
    assert dumped.startswith("[00:01.50]First\n")
    assert "[00:03.50]Second" in dumped
    assert "[01:05.50]Third" in dumped
    assert "[02:00.05]Fourth" in dumped


def test_srt_dump_renumbers_from_one(tmp_path):
    path = tmp_path / "a.srt"
    cues = [
        Cue(index=643, text="First", start=1.234, end=2.345),
        Cue(index=650, text="Second", start=3.0, end=4.0),
    ]
    srt.dump(path, cues)
    assert cues[0].index == 1
    assert cues[1].index == 2
    loaded = srt.load(path)
    assert [c.index for c in loaded] == [1, 2]
    text = path.read_text(encoding="utf-8")
    assert text.startswith("1\n")
    assert "\n2\n" in text


def test_lrc_dump_renumbers_cue_index(tmp_path):
    path = tmp_path / "a.lrc"
    cues = [
        Cue(index=643, text="First", start=1.0, end=3.5),
        Cue(index=650, text="Second", start=3.5, end=5.0),
    ]
    lrc.dump(path, cues)
    assert [c.index for c in cues] == [1, 2]
    loaded = lrc.load(path)
    assert [c.index for c in loaded] == [1, 2]
    assert [c.text for c in loaded] == ["First", "Second"]
