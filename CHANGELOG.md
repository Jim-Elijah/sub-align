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
