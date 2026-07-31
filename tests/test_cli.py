from __future__ import annotations

from sub_align.cli import build_parser, main


def test_cli_help():
    parser = build_parser()
    args = parser.parse_args(["media.mp4", "subs.srt", "--language", "en", "-o", "out.srt"])
    assert args.language == "en"
    assert str(args.output) == "out.srt"
    assert args.fill_gaps is False


def test_cli_fill_gaps_flag():
    parser = build_parser()
    args = parser.parse_args(["media.mp4", "subs.srt", "--language", "en", "--fill-gaps"])
    assert args.fill_gaps is True


def test_cli_trim_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["media.mp3", "script.txt", "--language", "en", "--trim-start", "13", "--trim-end", "5"]
    )
    assert args.trim_start == 13.0
    assert args.trim_end == 5.0


def test_cli_offset_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["media.mp3", "subs.srt", "--language", "en", "--offset", "12.5", "--no-auto-offset"]
    )
    assert args.offset == 12.5
    assert args.no_auto_offset is True


def test_cli_allows_audio_only():
    parser = build_parser()
    args = parser.parse_args(["media.mp3", "--language", "en"])
    assert str(args.media) == "media.mp3"
    assert args.subtitle is None
    assert args.max_words is None
    assert args.max_chars is None
    assert args.max_duration is None


def test_cli_max_line_limit_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "media.mp3",
            "--language",
            "en",
            "--max-words",
            "12",
            "--max-chars",
            "42",
            "--max-duration",
            "8",
        ]
    )
    assert args.max_words == 12
    assert args.max_chars == 42
    assert args.max_duration == 8.0


def test_cli_no_timings_flag():
    parser = build_parser()
    args = parser.parse_args(["media.mp3", "--language", "en", "--no-timings"])
    assert args.no_timings is True


def test_cli_missing_files_returns_error():
    code = main(["/no/such/media.wav", "--language", "en"])
    assert code == 1
