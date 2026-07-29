# Sub-align — align SRT/LRC/TXT subtitles to audio/video with WhisperX

Force-align existing subtitle text to media so cues track speech even when
the file has leading/trailing silence or local drift. Unlike tools that only
shift the whole timeline (e.g. ffsubsync), this package maps each cue onto
detected speech and then runs WhisperX phoneme alignment.

## Features

- Formats: `.srt`, `.lrc`, `.txt`
- Automatic strategy by input type:
  - `.txt`: Whisper transcription for rough time windows, then WhisperX forced
    align on the **original script lines** (not ASR wording)
  - `.srt` / `.lrc`: keep original timestamps, expand by a margin, then align
- Devices: `auto` / `cpu` / `cuda`
- CLI and Python API
- Packaged for PyPI (`sub-align`)

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH`
- Enough disk/RAM for WhisperX alignment models on first run

## Install

### From PyPI

```bash
pip install 'sub-align[align]'
# or
uv pip install 'sub-align[align]'
```

Extras `[align]`, `[cpu]`, and `[gpu]` all install WhisperX. Install a matching
PyTorch build first when you need a specific CPU/CUDA wheel:

**CPU torch, then align stack:**

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install 'sub-align[cpu]'
```

**CUDA torch, then align stack:**

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install 'sub-align[gpu]'
```

### Development (uv)

```bash
uv venv
uv sync --group dev          # unit tests / lint (no WhisperX)
uv sync --group dev --extra align   # full local alignment
uv run pytest
uv run ruff check src tests
```

## CLI

```bash
sub-align media.mp4 subs.srt --language zh -o out.srt
sub-align audio.wav lyrics.lrc --language en --margin 1.0
sub-align media.mkv script.txt --language en --model small
sub-align media.mkv subs.srt --detect-language --device cuda
```

| Flag | Meaning |
|------|---------|
| `--language` | Alignment model language (`en`, `zh`, …) |
| `--detect-language` | Use a tiny Whisper model when language is omitted |
| `--margin` | Search-window padding in seconds |
| `--device` | `auto`, `cpu`, or `cuda` |
| `--model` | Whisper model name or local path for `.txt` transcription windows (default: `small`) |
| `--compute-type` | Override default (`float16` on CUDA, `int8` on CPU) |
| `--fill-gaps` | Extend each cue end to the next cue start (last cue ends at audio duration) |

## Python API

```python
from sub_align import align_file

align_file(
    media="a.mp4",
    subtitle="a.srt",
    output="a.aligned.srt",
    language="zh",
    device="auto",
)
```

For `subtitle` input, strategy is chosen automatically:

- `.txt` inputs are transcribed only to estimate search windows; forced alignment
  uses each original script line as segment text
- `.srt` and `.lrc` inputs reuse their existing timestamps as alignment windows

## How it differs from ffsubsync

ffsubsync typically applies a global offset or linear stretch. With leading or
trailing silence that can slide the whole subtitle track incorrectly.
`sub-align` chooses its alignment strategy from the subtitle format: plain-text
`.txt` inputs are transcribed to estimate per-line search windows, then each
original script line is force-aligned inside those windows; `.srt` and
`.lrc` inputs reuse their existing timestamps as search windows. It then runs
WhisperX forced alignment so each line can move independently instead of
applying one global shift.

**Limitation:** subtitle text must roughly match spoken content. Alignment does
not correct wrong words.

## Build and publish

```bash
uv build
uv publish   # requires PyPI credentials / UV_PUBLISH_TOKEN
```

GitHub Actions publishes on tags matching `v*` (see `.github/workflows/publish.yml`).
Configure either Trusted Publishing on PyPI or repository secret `UV_PUBLISH_TOKEN`.

## License

MIT
