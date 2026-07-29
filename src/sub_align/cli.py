from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sub_align import __version__
from sub_align.align import align_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sub-align",
        description="Align LRC/SRT subtitles to audio/video with WhisperX forced alignment.",
    )
    parser.add_argument("media", type=Path, help="Audio or video file")
    parser.add_argument("subtitle", type=Path, help="Subtitle file (.srt or .lrc)")
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
        "--mode",
        default="realign",
        choices=["realign", "refine"],
        help="realign maps cues onto VAD speech; refine expands existing timestamps",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Seconds of padding around each search window (default: 0.5)",
    )
    parser.add_argument(
        "--vad-method",
        default="energy",
        choices=["energy", "silero"],
        help="Speech detection method for realign mode (default: energy)",
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
    try:
        out = align_file(
            media=args.media,
            subtitle=args.subtitle,
            output=args.output,
            language=args.language,
            detect_language=args.detect_language,
            device=args.device,
            compute_type=args.compute_type,
            mode=args.mode,
            margin=args.margin,
            vad_method=args.vad_method,
            print_progress=args.print_progress,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
