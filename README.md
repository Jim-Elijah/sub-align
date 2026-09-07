# Sub-align

[![CI Status](https://github.com/Jim-Elijah/sub-align/actions/workflows/ci.yml/badge.svg)](https://github.com/Jim-Elijah/sub-align/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/sub-align.svg)](https://pypi.org/project/sub-align)
[![PyPI Version](https://img.shields.io/pypi/v/sub-align.svg)](https://pypi.org/project/sub-align)

Align `.srt` / `.lrc` / `.txt` subtitles (or generate them) to audio/video with
[WhisperX](https://github.com/m-bain/whisperX) forced alignment — so each cue can
move independently instead of only applying one global timeline shift.

## Why not only a global offset?

Tools like [ffsubsync](https://github.com/smacke/ffsubsync) typically find a
constant offset (or stretch) between speech activity and subtitle “on” times.
That works well for whole-track drift, but leading/trailing silence or local
timing errors can still leave lines early or late.

`sub-align` picks a strategy from the input type, then runs **WhisperX
phoneme / word-level forced alignment** so each line is refined against the
audio:

| Input | Strategy |
|-------|----------|
| Media only | Whisper ASR → word-align → split into timed cues |
| `.txt` script | ASR only for search windows → forced-align **original lines** |
| `.srt` / `.lrc` | Optional global offset → expand windows by `--margin` → forced-align |

**Limitations:** subtitle text must roughly match spoken content (no
translation). Refine can still nudge already-good cues; very short lines may be
merged with neighbors for alignment; there is no speaker diarization. See
[docs/pipeline.md § Limitations](docs/pipeline.md#limitations).

More detail: [docs/pipeline.md](docs/pipeline.md) · scenarios & flags:
[docs/usage.md](docs/usage.md)

## Install

Requires **Python 3.10+** and [ffmpeg](https://ffmpeg.org/) on `PATH`. First run
downloads WhisperX alignment models (disk/RAM).

```bash
pip install 'sub-align[align]'
# or
uv pip install 'sub-align[align]'
```

Extras `[align]`, `[cpu]`, and `[gpu]` all install WhisperX. Install a matching
PyTorch build first when you need a specific CPU/CUDA wheel:

```bash
# CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install 'sub-align[cpu]'

# CUDA (example: cu124)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install 'sub-align[gpu]'
```

### Development

```bash
uv venv
uv sync --group dev                 # unit tests / lint (no WhisperX)
uv sync --group dev --extra align   # full local alignment
uv run pytest
uv run ruff check src tests
```

CI runs the same unit tests **without** the `align` extra. Path A/B/C coverage
also uses checked-in fixtures under `tests/fixtures/` (timed WAVs + recorded
WhisperX ASR/align JSON), replayed via mocks in `tests/test_fixture_replay.py`.

To regenerate those fixtures locally (needs ffmpeg; recording needs WhisperX):

```bash
# 1) Timed TTS clips (clip 1 = main coverage, clip 2 = dash dialogue)
uv run --with edge-tts --with numpy --with soundfile \
  scripts/generate_clip1_tts.py --clip 1
uv run --with edge-tts --with numpy --with soundfile \
  scripts/generate_clip1_tts.py --clip 2

# 2) Record ASR + word-align JSON for replay tests
uv run --extra align scripts/record_whisperx_fixtures.py \
  tests/fixtures/clip1_timed.wav tests/fixtures/clip2_timed.wav
```

Companion subtitle inputs (`clip1_script.txt`, `clip1_drift.srt`,
`clip2_dialogue.srt`) live beside the WAVs; edit those if the cue sheet changes.

## Usage

```bash
# Timed subtitles: auto global offset + per-cue refine
sub-align media.mp4 subs.srt --language zh -o out.srt

# Lyrics (LRC): same strategy as SRT
sub-align audio.wav lyrics.lrc --language en --margin 1.0

# Untimed script: ASR windows, then force-align original lines
sub-align media.mkv script.txt --language en --model small

# Skip podcast intro/outro before aligning a script
sub-align media.mp3 script.txt --language en --trim-start 13 --trim-end 5

# Known whole-track shift (skips auto-offset ASR)
sub-align media.mkv subs.srt --language en --offset 12.5

# Audio only: transcribe + word-align into an SRT
sub-align lecture.mp4 --language en -o lecture.asr.srt
```

Always pass `--language` (e.g. `en`, `zh`) or `--detect-language`.

See [docs/usage.md](docs/usage.md) for when to use `--model`, `--margin`,
`--offset`, `--fill-gaps`, `--trim-*`, audio-only line limits, and a Whisper
model size / VRAM cheat sheet.

## Python API

```python
from sub_align import align_file

align_file(
    media="a.mp4",
    subtitle="a.srt",   # omit for audio-only transcription
    output="a.aligned.srt",
    language="zh",
    device="auto",
)
```

## How it works (short)

1. Load media as 16 kHz mono audio (via WhisperX / ffmpeg); optional
   `--trim-start` / `--trim-end`.
2. Resolve language (`--language` or tiny-model detection).
3. Build search windows by input type (ASR token match for `.txt`; offset +
   margin refine for `.srt`/`.lrc`; full ASR for media-only).
4. Run WhisperX forced alignment; remap word times onto original cues; trim
   overlaps; optional `--fill-gaps`; write `.srt` or `.lrc`.

Full diagram and tech notes: [docs/pipeline.md](docs/pipeline.md).

## Build and publish

```bash
uv build
uv publish   # requires PyPI credentials / UV_PUBLISH_TOKEN
```

GitHub Actions publishes on tags matching `v*` (see `.github/workflows/publish.yml`).
Configure Trusted Publishing on PyPI or repository secret `UV_PUBLISH_TOKEN`.

## License

MIT
