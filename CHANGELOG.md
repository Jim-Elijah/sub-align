## Unreleased

### Fix

- refine: merge short ``.srt``/``.lrc`` cues before forced alignment only when
  the inter-cue pause is small (≲1.5s); large gaps keep short cues alone
- refine: keep original short-cue start when FA moves later (avoids swallowed
  onsets on Yes/Okay-style lines)
- refine: keep original short-cue times when FA collapses or shrinks them badly
- refine: restore cues left at ``start == end`` after VAD clamp / overlap trim
- refine: VAD start clamp only blocks FA pulling *earlier* into silence; never
  snaps start later than the timed cue (energy VAD soft-onset misses)
- refine: temporarily split multi-line ``-A / -B [/ -C…]`` dialogue for FA, then merge
- refine: use a smaller start-side search margin (capped at 0.25s) to reduce lead-in drift

## v0.2.0 (2026-08-09)

### Feat

- add pipeline step timing logs to stderr
- support audio-only transcription with optional cue length limits
- add media trim and global offset for .srt/.lrc refine
- align .txt via ASR windows on original script lines
- support plain text subtitle format in alignment
- add sub-align for WhisperX LRC/SRT forced alignment

### Fix

- renumber cue indices to 1..n on SRT/LRC dump
