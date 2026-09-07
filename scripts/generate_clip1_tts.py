#!/usr/bin/env python3
"""Generate timed fixture WAVs (clip1 / clip2) from English cue sheets.

Dev-only helper (not wired into CI or default pytest). Each spoken line is
synthesized separately with edge-tts, then placed on an absolute timeline so
cue starts match the sheet. Overruns are handled by trimming trailing silence,
slight speed-up, or shifting later cues while preserving long silence islands.

Usage::

    # from repo root — Clip 1 (main coverage, ~42–43 s)
    uv run --with edge-tts --with numpy --with soundfile \\
        scripts/generate_clip1_tts.py --clip 1

    # Clip 2 (dash dialogue, ~20 s)
    uv run --with edge-tts --with numpy --with soundfile \\
        scripts/generate_clip1_tts.py --clip 2

    # optional flags
    uv run --with edge-tts --with numpy --with soundfile \\
        scripts/generate_clip1_tts.py --clip 2 --voice en-US-JennyNeural \\
        -o tests/fixtures/clip2_timed.wav

Requires ``ffmpeg`` on PATH (decodes edge-tts MP3 → mono PCM).

WhisperX follow-up (optional, not run here)::

    # Record ASR + word-align JSON for fixture replay tests:
    #   uv run --extra align scripts/record_whisperx_fixtures.py \\
    #       tests/fixtures/clip1_timed.wav tests/fixtures/clip2_timed.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
VOICE_DEFAULT = "en-US-JennyNeural"
MAX_SPEEDUP = 1.35
REPO_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("generate_clip_tts")


@dataclass(frozen=True)
class Cue:
    start: float
    text: str
    """If set, synthesize each part and join with ``gap_s`` silence (numbers)."""
    parts: tuple[str, ...] | None = None
    gap_s: float = 0.18


@dataclass(frozen=True)
class ClipSpec:
    name: str
    out_name: str
    cues: tuple[Cue, ...]
    silence_islands: tuple[tuple[float, float], ...]
    """Ideal end of last utterance / timeline anchor before trailing silence."""
    end_anchor_s: float
    trailing_silence_s: float
    """Soft expected duration band for CLI warning (inclusive)."""
    duration_lo: float
    duration_hi: float

    @property
    def default_out(self) -> Path:
        return REPO_ROOT / "tests" / "fixtures" / self.out_name


CLIP1 = ClipSpec(
    name="clip1",
    out_name="clip1_timed.wav",
    cues=(
        Cue(1.5, "Hello."),
        Cue(3.5, "Alright, that's it."),
        Cue(8.0, "We need more time to finish the work today."),
        Cue(13.5, "Yes."),
        Cue(15.5, "The quick brown fox jumps over the lazy dog near the river."),
        Cue(24.0, "Okay."),
        Cue(26.0, "Please wait here until I come back."),
        Cue(32.0, "One. Two. Three.", parts=("One.", "Two.", "Three."), gap_s=0.22),
        Cue(38.0, "That's all for now."),
    ),
    silence_islands=(
        (0.0, 1.5),
        (6.0, 8.0),
        (22.0, 24.0),
        (30.0, 32.0),
        (36.0, 38.0),
    ),
    end_anchor_s=41.0,
    trailing_silence_s=1.5,
    duration_lo=41.0,
    duration_hi=45.0,
)

# Clip 2 — dash dialogue (~20 s). Spoken text omits leading "- " (SRT keeps dashes).
CLIP2 = ClipSpec(
    name="clip2",
    out_name="clip2_timed.wav",
    cues=(
        Cue(1.0, "Where are you going?"),
        Cue(3.5, "I'm going home now."),
        Cue(7.5, "Can you wait a minute?"),
        Cue(10.5, "No, I really have to leave."),
        Cue(15.0, "Fine. See you tomorrow."),
    ),
    silence_islands=(
        (0.0, 1.0),
        (6.5, 7.5),
        (14.0, 15.0),
    ),
    end_anchor_s=18.5,
    trailing_silence_s=1.5,
    duration_lo=18.0,
    duration_hi=22.0,
)

CLIPS: dict[str, ClipSpec] = {"1": CLIP1, "2": CLIP2, "clip1": CLIP1, "clip2": CLIP2}


def _mp3_to_mono_f32(mp3: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode MP3 bytes to mono float32 PCM via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=mp3,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode TTS audio: {err or proc.returncode}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg returned empty PCM for TTS audio")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


async def _tts_mp3(text: str, voice: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError(f"edge-tts returned no audio for: {text!r}")
    return b"".join(chunks)


async def synthesize_line(cue: Cue, voice: str) -> np.ndarray:
    if cue.parts:
        segments: list[np.ndarray] = []
        gap = np.zeros(int(round(cue.gap_s * SAMPLE_RATE)), dtype=np.float32)
        for i, part in enumerate(cue.parts):
            mp3 = await _tts_mp3(part, voice)
            segments.append(_mp3_to_mono_f32(mp3))
            if i < len(cue.parts) - 1:
                segments.append(gap)
        return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

    mp3 = await _tts_mp3(cue.text, voice)
    return _mp3_to_mono_f32(mp3)


def trim_trailing_silence(
    audio: np.ndarray,
    *,
    threshold: float = 0.012,
    pad_s: float = 0.04,
) -> np.ndarray:
    """Drop near-silent tail; keep a short pad after the last active sample."""
    if audio.size == 0:
        return audio
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio[: max(1, int(pad_s * SAMPLE_RATE))]
    end = min(audio.size, int(active[-1]) + 1 + int(round(pad_s * SAMPLE_RATE)))
    return audio[:end]


def speed_up(audio: np.ndarray, factor: float) -> np.ndarray:
    """Linear-resample speed-up (factor > 1 → shorter)."""
    if factor <= 1.0 or audio.size < 2:
        return audio
    n_out = max(1, int(round(audio.size / factor)))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def island_start_after(t: float, islands: tuple[tuple[float, float], ...]) -> float | None:
    for a, _b in islands:
        if a > t + 1e-9:
            return a
    return None


def fit_utterance(
    audio: np.ndarray,
    *,
    start: float,
    deadline: float,
    label: str,
) -> np.ndarray:
    """Fit ``audio`` into ``[start, deadline)`` via trim then speed-up."""
    budget = deadline - start
    if budget <= 0:
        raise ValueError(f"non-positive budget for {label!r} at {start}")

    fitted = trim_trailing_silence(audio)
    dur = fitted.size / SAMPLE_RATE
    if dur <= budget:
        return fitted

    trimmed_dur = dur
    factor = dur / budget
    if factor <= MAX_SPEEDUP:
        fitted = speed_up(fitted, factor)
        log.warning(
            "sped up %r by %.2fx (%.2fs → ≤%.2fs budget ending %.2f)",
            label,
            factor,
            trimmed_dur,
            budget,
            deadline,
        )
        return fitted

    fitted = speed_up(fitted, MAX_SPEEDUP)
    log.warning(
        "sped up %r by %.2fx but still overruns deadline %.2f (need shift)",
        label,
        MAX_SPEEDUP,
        deadline,
    )
    return fitted


def resolve_starts(
    raw_starts: list[float],
    durations: list[float],
    *,
    islands: tuple[tuple[float, float], ...],
    end_anchor_s: float,
) -> list[float]:
    """
    Ensure each utterance ends before the next start / next island.

    Later cues may shift forward; if a shift would land inside a silence island,
    jump to the island end. Logs warnings on any adjustment.
    """
    starts = list(raw_starts)
    n = len(starts)
    for i in range(n):
        dur = durations[i]
        end = starts[i] + dur

        if i + 1 < n:
            limit = starts[i + 1]
        else:
            limit = max(end_anchor_s, end)
        island = island_start_after(starts[i], islands)
        if island is not None:
            limit = min(limit, island)

        if end <= limit + 1e-6:
            continue

        overrun = end - limit
        if i + 1 < n:
            log.warning(
                "shifting cues after index %d by +%.3fs to clear overrun",
                i,
                overrun,
            )
            for j in range(i + 1, n):
                starts[j] += overrun
                for a, b in islands:
                    if a < starts[j] < b:
                        log.warning(
                            "cue %d start %.3f fell inside island [%.1f, %.1f); "
                            "snapping to %.1f",
                            j,
                            starts[j],
                            a,
                            b,
                            b,
                        )
                        starts[j] = b
        else:
            log.warning(
                "last utterance extends past %.2fs (ends %.2fs); keeping speech",
                limit,
                end,
            )
    return starts


def build_timeline(
    clips: list[np.ndarray],
    starts: list[float],
    *,
    end_anchor_s: float,
    trailing_silence_s: float,
) -> np.ndarray:
    ends = [s + c.size / SAMPLE_RATE for s, c in zip(starts, clips, strict=True)]
    total_s = max(max(ends), end_anchor_s) + trailing_silence_s
    n = int(round(total_s * SAMPLE_RATE))
    timeline = np.zeros(n, dtype=np.float32)
    for start, clip in zip(starts, clips, strict=True):
        i0 = int(round(start * SAMPLE_RATE))
        i1 = i0 + clip.size
        if i1 > timeline.size:
            pad = i1 - timeline.size
            timeline = np.concatenate([timeline, np.zeros(pad, dtype=np.float32)])
        timeline[i0:i1] += clip
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1.0:
        timeline *= 0.99 / peak
        log.warning("normalized timeline peak %.3f → 0.99", peak)
    return timeline


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError:
        import wave

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_i16.tobytes())
        return

    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


async def generate(spec: ClipSpec, voice: str, out: Path) -> float:
    cues = spec.cues
    islands = spec.silence_islands
    log.info("synthesizing %s (%d cues) voice=%s", spec.name, len(cues), voice)
    raw_clips = [await synthesize_line(cue, voice) for cue in cues]

    nominal = [c.start for c in cues]
    fitted: list[np.ndarray] = []
    for i, (cue, clip) in enumerate(zip(cues, raw_clips, strict=True)):
        next_start = cues[i + 1].start if i + 1 < len(cues) else spec.end_anchor_s
        island = island_start_after(cue.start, islands)
        deadline = next_start if island is None else min(next_start, island)
        fitted.append(
            fit_utterance(clip, start=cue.start, deadline=deadline, label=cue.text)
        )

    durations = [c.size / SAMPLE_RATE for c in fitted]
    starts = resolve_starts(
        nominal,
        durations,
        islands=islands,
        end_anchor_s=spec.end_anchor_s,
    )
    for cue, start, dur in zip(cues, starts, durations, strict=True):
        if abs(start - cue.start) > 1e-3:
            log.warning(
                "cue %r start adjusted %.3f → %.3f (dur=%.3fs)",
                cue.text,
                cue.start,
                start,
                dur,
            )
        else:
            log.info("cue %r @ %.3fs (dur=%.3fs)", cue.text, start, dur)

    timeline = build_timeline(
        fitted,
        starts,
        end_anchor_s=spec.end_anchor_s,
        trailing_silence_s=spec.trailing_silence_s,
    )
    write_wav(out, timeline)
    return timeline.size / SAMPLE_RATE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--clip",
        choices=sorted({*CLIPS.keys()}),
        default="1",
        help="which cue sheet to render (default: 1)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output WAV path (default: tests/fixtures/clipN_timed.wav)",
    )
    p.add_argument(
        "--voice",
        default=VOICE_DEFAULT,
        help=f"edge-tts voice (default: {VOICE_DEFAULT})",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    spec = CLIPS[args.clip]
    out = args.output if args.output is not None else spec.default_out

    try:
        duration = asyncio.run(generate(spec, args.voice, out))
    except FileNotFoundError as exc:
        log.error("missing dependency: %s (is ffmpeg installed?)", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        log.error("%s", exc)
        return 1

    out_r = out.resolve()
    exists = out_r.is_file()
    size = out_r.stat().st_size if exists else 0
    print(f"wrote {out_r}")
    print(f"exists={exists} size_bytes={size} duration_s={duration:.3f}")
    if not exists:
        return 1
    if duration < spec.duration_lo or duration > spec.duration_hi:
        log.warning(
            "duration %.3fs outside expected %.1f–%.1fs band for %s",
            duration,
            spec.duration_lo,
            spec.duration_hi,
            spec.name,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
