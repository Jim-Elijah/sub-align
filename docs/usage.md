# Usage guide

Quick CLI patterns, when to pick which input type, and flag reference. For the
alignment algorithm itself, see [pipeline.md](pipeline.md).

## Choosing an input

| Situation | Use |
|-----------|-----|
| No script; need timed subtitles from speech | Media only |
| Correct line-by-line script, no reliable times | `.txt` |
| Existing SRT/LRC with whole-track drift | `.srt` / `.lrc` (default auto-offset) |
| Times mostly OK; need local mouth-sync refine | `.srt` / `.lrc` + `--no-auto-offset` |
| Known constant delay (e.g. +12.5s) | `.srt` / `.lrc` + `--offset` |
| Continuous on-screen lyrics / practice cues | add `--fill-gaps` |
| Podcast intro not in the script | `--trim-start` / `--trim-end` |

Always pass `--language` (or `--detect-language`). Text must roughly match
speech when aligning a subtitle file. For known accuracy trade-offs (short cues,
refine drift, VAD, no diarization), see
[pipeline.md § Limitations](pipeline.md#limitations).

---

## Scenario examples

### 1. Audio / video only

Generate an SRT from speech (ASR text + word-level timing).

```bash
sub-align lecture.mp4 --language zh -o lecture.asr.srt
sub-align podcast.mp3 --language en --model medium \
  --max-words 12 --max-chars 42 --max-duration 8
```

- Default cue splits: strong punctuation only (good for sentence-length lines).
- Add `--max-words` / `--max-chars` / `--max-duration` for shorter display lines.
- `--margin` and `--offset` do nothing here.

### 2. Media + `.txt` script

One non-empty line per cue. Output keeps script wording.

```bash
sub-align talk.mkv script.txt --language en --model small
sub-align episode.mp3 script.txt --language en \
  --trim-start 13 --trim-end 5 --margin 1.0
```

- ASR builds search windows; WhisperX force-aligns **original lines**.
- `--offset` / `--no-auto-offset` are ignored.
- Noisy audio or weak matching → try a larger `--model` and/or larger `--margin`.

### 3. Media + `.srt`

```bash
# Auto-estimate global offset, then refine each cue
sub-align movie.mkv movie.srt --language zh

# You already measured the shift
sub-align movie.mkv movie.srt --language en --offset 12.5

# Timeline is close; refine only
sub-align movie.mkv movie.srt --language en --no-auto-offset --margin 0.8
```

### 4. Media + `.lrc`

Same strategy as SRT (offset + margin refine + forced align). Ends are inferred
from the next line’s start when loading.

```bash
sub-align song.wav lyrics.lrc --language en --margin 1.0
sub-align song.mp3 lyrics.lrc --language ja --offset 0.8 --fill-gaps
```

Larger `--margin` helps with gaps between sung lines; `--fill-gaps` extends each
line until the next starts (useful for karaoke-style continuous display).

---

## Parameter reference

| Flag | Default | Applies to | Meaning |
|------|---------|------------|---------|
| `--language` | — | all | Alignment / ASR language (`en`, `zh`, …) |
| `--detect-language` | off | all | Tiny Whisper when `--language` omitted |
| `--model` | `small` | audio-only, `.txt`, auto-offset | Whisper ASR model name or path |
| `--margin` | `0.5` | `.txt`, `.srt`, `.lrc` | Seconds of padding on each search window |
| `--offset` | unset | `.srt`, `.lrc` | Constant cue shift (seconds); skips auto-offset |
| `--no-auto-offset` | off | `.srt`, `.lrc` | Do not estimate global offset via ASR |
| `--trim-start` | `0` | all | Drop leading seconds before alignment |
| `--trim-end` | `0` | all | Drop trailing seconds before alignment |
| `--fill-gaps` | off | all | Extend each cue end to the next start (last → audio end) |
| `--device` | `auto` | all | `auto`, `cpu`, or `cuda` |
| `--compute-type` | device default | all | e.g. `float16` (CUDA), `int8` (CPU) |
| `--max-words` | unset | audio-only | Cap words per cue after punctuation split |
| `--max-chars` | unset | audio-only | Cap characters per cue after punctuation split |
| `--max-duration` | unset | audio-only | Cap seconds per cue after punctuation split |
| `-o` / `--output` | auto name | all | Output `.srt` or `.lrc` path |
| `--print-progress` | off | all | WhisperX align progress |
| `--no-timings` | off | all | Hide per-step timing on stderr |

### When to tune common flags

**`--model`** — Larger models help hard accents, noise, weak `.txt` matching, or
bad auto-offsets. They do **not** change the WhisperX aligner weights (those
follow `--language`). Default `small` is a good balance for most runs.

Approximate Whisper sizes and cost (order-of-magnitude; depends on
implementation, batch size, and compute type):

| Model | Parameters | VRAM (GPU) | Relative speed |
|-------|------------|------------|----------------|
| `tiny` | 39M | ~1 GB | ~10× |
| `base` | 74M | ~1 GB | ~7× |
| `small` | 244M | ~2 GB | ~4× |
| `medium` | 769M | ~5 GB | ~2× |
| `large` / `large-v2` / `large-v3` | 1550M | ~10 GB | 1× |
| `turbo` | 809M | ~6 GB | ~8× |

Hardware notes:

- **GPU** — NVIDIA + CUDA recommended. Rough VRAM floors: `large` ≥10 GB,
  `medium` ≥5 GB; `small` and below usually fit entry-level cards.
- **CPU** — Works without a GPU but is much slower (`large` can be ~10–50×).
- **RAM** — Aim for ≥8 GB system memory; prefer ≥16 GB for `large`.
- **Disk** — Models cache locally (~75 MB for `tiny`, ~3 GB for `large`).

Rule of thumb: `tiny` / `base` for low-end machines or quick smoke tests;
`small` / `medium` for routine quality; `large` / `turbo` when accuracy
matters and you have a capable GPU.

**`--margin`** — `0.3–0.5` when SRT times are already close; `1.0–1.5` when local
drift is large or LRC timing is loose. Very large margins increase overlapping
search windows (later overlap trim / word remap still apply).

**`--offset` / `--no-auto-offset`** — Prefer `--offset` when you already know the
global delay (faster, no ASR). Prefer `--no-auto-offset` when the track is
mostly right or ASR would be unreliable. Leave the default for typical
whole-track drift.

**`--trim-start` / `--trim-end`** — Script or subs start at the “content”
while media has an intro/outro. Output times remain on the original media clock.
`trim_start + trim_end` must be less than duration.

**`--fill-gaps`** — Continuous display (karaoke, dense practice cues). Skip when
you want natural pauses between lines.

**`--max-words` / `--max-chars` / `--max-duration`** — Audio-only display-oriented
lines. Suggested starting points: `12` / `42` / `8`. Ignored for `.txt`/`.srt`/`.lrc`.

---

## Python API

```python
from sub_align import align_file

align_file(
    media="a.mp4",
    subtitle="a.srt",  # None → audio-only
    output="a.aligned.srt",
    language="zh",
    device="auto",
    margin=0.5,
    model_name="small",
    fill_gaps=False,
    trim_start=0.0,
    trim_end=0.0,
    offset=None,       # or e.g. 12.5
    auto_offset=True,
    max_words=None,
    max_chars=None,
    max_duration=None,
)
```

CLI maps `--model` → `model_name`, and `--no-auto-offset` → `auto_offset=False`.
