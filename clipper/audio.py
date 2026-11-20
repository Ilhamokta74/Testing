from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .util import run

GRID = 0.5  # seconds per signal sample


def extract_wav(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _norm(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    p = float(np.percentile(a, 95))
    if p <= 1e-9:
        return np.zeros_like(a)
    return np.clip(a / p, 0.0, 1.0)


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or a.size == 0:
        return a
    return np.convolve(a, np.ones(k) / k, mode="same")


def _local_z(a: np.ndarray, win: int = 120) -> np.ndarray:
    """z-score of each point against the local (about 60 s) mean/std."""
    pad = np.pad(a, (win // 2, win // 2), mode="edge")
    c1 = np.cumsum(np.insert(pad, 0, 0.0))
    c2 = np.cumsum(np.insert(pad ** 2, 0, 0.0))
    n = a.size
    s1 = c1[win:win + n] - c1[:n]
    s2 = c2[win:win + n] - c2[:n]
    mean = s1 / win
    var = np.maximum(s2 / win - mean ** 2, 1e-9)
    return (a - mean) / np.sqrt(var)


def compute_signals(wav_path: Path, words: list[dict], duration: float,
                    heatmap: list[dict] | None, grid: float = GRID) -> dict:
    sr = 16000
    x = read_wav(wav_path)
    hop = int(sr * grid)
    n = max(1, math.ceil(duration / grid))
    x = np.pad(x, (0, max(0, n * hop - x.size)))[: n * hop]
    frames = x.reshape(n, hop)

    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    energy = _norm(np.maximum(_local_z(db), 0.0))

    flat = np.zeros(n, dtype=np.float32)
    for i in range(0, n, 400):
        spec = np.abs(np.fft.rfft(frames[i:i + 400], axis=1)) ** 2 + 1e-12
        flat[i:i + 400] = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    burst = np.maximum(_local_z(db) - 1.0, 0.0)          # sudden loudness
    broadband = np.clip((flat - np.percentile(flat, 60)) /
                        (np.percentile(flat, 95) - np.percentile(flat, 60) + 1e-9), 0, 1)
    laughter = _norm(_smooth(burst * broadband, 3))

    counts = np.zeros(n, dtype=np.float32)
    for w in words:
        idx = int(((w["start"] + w["end"]) / 2) / grid)
        if 0 <= idx < n:
            counts[idx] += 1
    speech_rate = _norm(_smooth(counts, 10))

    hm = None
    if heatmap:
        mids = np.array([(h["start_time"] + h["end_time"]) / 2 for h in heatmap])
        vals = np.array([h["value"] for h in heatmap], dtype=np.float32)
        centers = (np.arange(n) + 0.5) * grid
        hm = _norm(np.interp(centers, mids, vals))

    return {
        "grid": grid, "n": n,
        "audio_energy": energy.round(4).tolist(),
        "laughter": laughter.round(4).tolist(),
        "speech_rate": speech_rate.round(4).tolist(),
        "heatmap": hm.round(4).tolist() if hm is not None else None,
    }


def interest_curve(signals: dict, weights: dict) -> np.ndarray:
    """Weighted combination of the signals; missing signals are dropped and weights renormalized."""
    total, acc = 0.0, np.zeros(signals["n"], dtype=np.float32)
    for k in ("audio_energy", "laughter", "speech_rate", "heatmap"):
        arr, w = signals.get(k), float(weights.get(k, 0))
        if arr is None or w <= 0:
            continue
        acc += w * np.asarray(arr, dtype=np.float32)
        total += w
    return acc / total if total > 0 else acc


def span_mean(arr, start: float, end: float, grid: float = GRID) -> float:
    if arr is None:
        return 0.0
    a = np.asarray(arr)
    i0, i1 = int(start / grid), max(int(start / grid) + 1, int(math.ceil(end / grid)))
    seg = a[i0:i1]
    return float(seg.mean()) if seg.size else 0.0
