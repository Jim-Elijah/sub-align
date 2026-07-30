from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sub_align import __version__
from sub_align.align import align_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sub-align",
        description="Align SRT/LRC/TXT subtitles to audio/video with WhisperX forced alignment.",
    )
    parser.add_argument("media", type=Path, help="Audio or video file")
    parser.add_argument(
        "subtitle",
        type=Path,
        nargs="?",
        default=None,
        help="Optional subtitle file (.srt, .lrc, or .txt) for forced alignment",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output subtitle path (default: <name>.aligned.<ext>)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for the alignment model (e.g. en, zh, ja)",
    )
    parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Detect language with a tiny Whisper model when --language is omitted",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2/Whisper compute type (default: float16 on CUDA, int8 on CPU)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Seconds of padding around each search window (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help=(
            "Whisper model name or local path for ASR and .txt windows and "
            ".srt/.lrc auto-offset (default: small)"
        ),
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Extend each cue end to the next cue start (last cue ends at audio duration)",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        help="Seconds to drop from the start of the media before alignment (default: 0)",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=0.0,
        help="Seconds to drop from the end of the media before alignment (default: 0)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help=(
            "Constant seconds to shift .srt/.lrc cues before refine "
            "(skips auto-offset; ignored for .txt)"
        ),
    )
    parser.add_argument(
        "--no-auto-offset",
        action="store_true",
        help="Disable automatic global offset estimation for .srt/.lrc",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Audio-only: max words per cue after punctuation split "
            "(e.g. 12); omit to keep full sentences"
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Audio-only: max characters per cue after punctuation split "
            "(e.g. 42); omit to keep full sentences"
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Audio-only: max seconds per cue after punctuation split "
            "(e.g. 8); omit to keep full sentences"
        ),
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Print WhisperX alignment progress",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_words is not None and args.max_words < 1:
        parser.error("--max-words must be >= 1")
    if args.max_chars is not None and args.max_chars < 1:
        parser.error("--max-chars must be >= 1")
    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be > 0")
    try:
        out = align_file(
            media=args.media,
            subtitle=args.subtitle,
            output=args.output,
            language=args.language,
            detect_language=args.detect_language,
            device=args.device,
            compute_type=args.compute_type,
            model_name=args.model,
            margin=args.margin,
            fill_gaps=args.fill_gaps,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            offset=args.offset,
            auto_offset=not args.no_auto_offset,
            print_progress=args.print_progress,
            max_words=args.max_words,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
