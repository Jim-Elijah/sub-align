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


def test_cli_missing_files_returns_error():
    code = main(["/no/such/media.wav", "/no/such.srt", "--language", "en"])
    assert code == 1
