#!/usr/bin/env python3
"""Record WhisperX ASR + word-align JSON fixtures from WAV files.

Dev-only helper (not wired into CI). Runs the same rough pipeline shape that
``align_file`` uses for timing anchors: ``transcribe`` then ``whisperx.align``
on ASR segments. Output is suitable for replaying in mock tests.

Usage::

    # from repo root (needs the align extra + ffmpeg)
    uv run --extra align scripts/record_whisperx_fixtures.py \\
        tests/fixtures/clip1_timed.wav

    uv run --extra align scripts/record_whisperx_fixtures.py \\
        tests/fixtures/clip1_timed.wav tests/fixtures/clip2_timed.wav \\
        --model tiny --language en --device cpu

    # default: both clip1/clip2 under tests/fixtures/ if present
    uv run --extra align scripts/record_whisperx_fixtures.py

Writes ``<stem>.whisperx.json`` next to each WAV (or ``-o`` for a single file).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DEFAULT_WAVS = (
    FIXTURES / "clip1_timed.wav",
    FIXTURES / "clip2_timed.wav",
)

log = logging.getLogger("record_whisperx_fixtures")


def _jsonable(obj: Any) -> Any:
    """Convert numpy / Path values so ``json.dump`` succeeds."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy scalars / arrays
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def record_one(
    media: Path,
    *,
    language: str,
    model_name: str,
    device: str,
    compute_type: str | None,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is required. Install with: uv sync --extra align "
            "(or: uv run --extra align ...)"
        ) from exc

    from sub_align.audio import load_audio
    from sub_align.device import default_compute_type, resolve_device

    resolved = resolve_device(device)
    ctype = default_compute_type(resolved, compute_type)

    log.info("loading audio %s", media)
    audio = load_audio(media)
    duration_s = float(len(audio) / 16_000)

    log.info("load_model name=%s device=%s compute_type=%s", model_name, resolved, ctype)
    model = whisperx.load_model(model_name, resolved, compute_type=ctype, language=language)

    log.info("transcribe language=%s batch_size=%d", language, batch_size)
    transcription = model.transcribe(audio, language=language, batch_size=batch_size)
    asr_segments = list(transcription.get("segments") or [])
    log.info("ASR segments: %d", len(asr_segments))

    asr_aligned: dict[str, Any] = {"segments": [], "word_segments": []}
    if asr_segments:
        log.info("load_align_model language=%s", language)
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=resolved,
        )
        log.info("align ASR segments")
        asr_aligned = whisperx.align(
            asr_segments,
            align_model,
            metadata,
            audio,
            resolved,
            return_char_alignments=False,
            print_progress=False,
        )

    return {
        "meta": {
            "media": str(media.resolve()),
            "media_name": media.name,
            "language": language,
            "model": model_name,
            "device": resolved,
            "compute_type": ctype,
            "sample_rate": 16_000,
            "duration_s": duration_s,
            "batch_size": batch_size,
            "note": (
                "asr = whisper model.transcribe(); "
                "asr_aligned = whisperx.align(asr segments). "
                "Replay these in tests instead of calling WhisperX."
            ),
        },
        "asr": {
            "language": transcription.get("language", language),
            "segments": asr_segments,
        },
        "asr_aligned": {
            "segments": asr_aligned.get("segments") or [],
            "word_segments": asr_aligned.get("word_segments") or [],
        },
    }


def default_out_for(media: Path) -> Path:
    # clip1_timed.wav → clip1_timed.whisperx.json
    return media.with_name(f"{media.stem}.whisperx.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "media",
        nargs="*",
        type=Path,
        help="WAV/media paths (default: clip1/clip2 fixtures if they exist)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output JSON path (only valid with a single media file)",
    )
    p.add_argument("--language", default="en")
    p.add_argument("--model", default="tiny", help="Whisper model name (default: tiny)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda (default: auto)")
    p.add_argument("--compute-type", default=None, help="override whisperx compute_type")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    media_list = list(args.media)
    if not media_list:
        media_list = [p for p in DEFAULT_WAVS if p.is_file()]
        if not media_list:
            log.error(
                "no media given and no default fixtures found under %s",
                FIXTURES,
            )
            return 1

    if args.output is not None and len(media_list) != 1:
        log.error("--output requires exactly one media file")
        return 1

    for media in media_list:
        if not media.is_file():
            log.error("media not found: %s", media)
            return 1
        out = args.output if args.output is not None else default_out_for(media)
        try:
            payload = record_one(
                media,
                language=args.language,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
            )
        except Exception as exc:  # noqa: BLE001 — CLI boundary
            log.error("%s", exc)
            return 1

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        n_asr = len(payload["asr"]["segments"])
        n_aligned = len(payload["asr_aligned"]["segments"])
        n_words = sum(len(s.get("words") or []) for s in payload["asr_aligned"]["segments"])
        if not n_words:
            n_words = len(payload["asr_aligned"]["word_segments"])
        print(f"wrote {out.resolve()}")
        print(
            f"  asr_segments={n_asr} aligned_segments={n_aligned} "
            f"words≈{n_words} duration_s={payload['meta']['duration_s']:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
