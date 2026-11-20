from __future__ import annotations

import re

TERMINAL = re.compile(r"[.!?…]+[\"')\]]*$")


def build_sentences(words: list[dict], gap: float = 0.6, max_words: int = 40) -> list[dict]:
    """Group words into numbered sentences (id = index + 1) by punctuation and pauses."""
    sentences: list[dict] = []
    cur: list[int] = []

    def flush():
        if not cur:
            return
        ws = [words[i] for i in cur]
        sentences.append({
            "id": len(sentences) + 1, "start": ws[0]["start"], "end": ws[-1]["end"],
            "w0": cur[0], "w1": cur[-1],
            "text": " ".join(w["w"] for w in ws),
        })
        cur.clear()

    for i, w in enumerate(words):
        if cur and w["start"] - words[cur[-1]]["end"] > gap:
            flush()
        cur.append(i)
        if TERMINAL.search(w["w"]) or len(cur) >= max_words:
            flush()
    flush()
    return sentences


def make_windows(sentences: list[dict], window_s: float, overlap_s: float) -> list[list[int]]:
    """Windows of sentence indices (0-based positions) ~window_s long with overlap."""
    if not sentences:
        return []
    windows: list[list[int]] = []
    t0, total = sentences[0]["start"], sentences[-1]["end"]
    step = max(10.0, window_s - overlap_s)
    t = t0
    while t < total:
        idx = [i for i, s in enumerate(sentences) if s["end"] > t and s["start"] < t + window_s]
        if idx and (not windows or idx != windows[-1]):
            windows.append(idx)
        if t + window_s >= total:
            break
        t += step
    return windows
