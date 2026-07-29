from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000


def _merge_spans(
    spans: list[tuple[float, float]],
    merge_gap: float = 0.3,
) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: item[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def energy_speech_spans(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_ratio: float = 0.2,
    min_speech_ms: float = 200.0,
) -> list[tuple[float, float]]:
    """Lightweight RMS energy VAD (no model download). Used as default/fallback."""
    if audio.size == 0:
        return []
    wave = audio.astype(np.float32, copy=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)

    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if wave.size < frame:
        duration = wave.size / sample_rate
        return [(0.0, duration)] if duration > 0 else []

    energies: list[float] = []
    for start in range(0, wave.size - frame + 1, hop):
        chunk = wave[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk))))
    energy = np.asarray(energies, dtype=np.float32)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 1e-8:
        return []

    threshold = peak * threshold_ratio
    speech = energy >= threshold
    min_frames = max(1, int(min_speech_ms / hop_ms))

    spans: list[tuple[float, float]] = []
    i = 0
    while i < speech.size:
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < speech.size and speech[j]:
            j += 1
        if j - i >= min_frames:
            start_t = (i * hop) / sample_rate
            end_t = min(wave.size / sample_rate, (j * hop + frame) / sample_rate)
            spans.append((start_t, end_t))
        i = j
    return _merge_spans(spans)


def silero_speech_spans(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Silero VAD via torch.hub. Falls back to energy VAD on failure."""
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]
        wave = audio.astype(np.float32, copy=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)
        tensor = torch.from_numpy(wave)
        timestamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
        )
        spans = [(item["start"] / sample_rate, item["end"] / sample_rate) for item in timestamps]
        return _merge_spans(spans)
    except Exception:
        return energy_speech_spans(audio, sample_rate=sample_rate)


def detect_speech_spans(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    method: str = "energy",
) -> list[tuple[float, float]]:
    """
    Detect speech regions.

    method:
      - energy: local RMS VAD (default; no download, CI-friendly)
      - silero: Silero VAD with energy fallback
    """
    if method == "silero":
        return silero_speech_spans(audio, sample_rate=sample_rate)
    if method == "energy":
        return energy_speech_spans(audio, sample_rate=sample_rate)
    raise ValueError(f"Unknown VAD method: {method!r}")
