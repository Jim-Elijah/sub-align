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


def test_cli_missing_files_returns_error():
    code = main(["/no/such/media.wav", "/no/such.srt", "--language", "en"])
    assert code == 1
