from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter

import numpy as np

from .audio import GRID, interest_curve, span_mean
from .gpu import GPU
from .util import ClipperError

SCORE_KEYS = ["hook", "emotion", "self_contained", "punchline", "ending_complete"]

CAND_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "start_sentence": {"type": "integer"},
            "end_sentence": {"type": "integer"},
            "scores": {"type": "object",
                       "properties": {k: {"type": "integer"} for k in SCORE_KEYS},
                       "required": SCORE_KEYS},
            "hook_sentence": {"type": "integer"},
            "use_teaser": {"type": "boolean"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["start_sentence", "end_sentence", "scores", "hook_sentence",
                     "use_teaser", "title", "description", "hashtags", "reason"],
    }}},
    "required": ["candidates"],
}

SYSTEM = (
    "You are an editor who picks the best short-form vertical clips (TikTok/Reels/Shorts) "
    "from a transcript. Sentences are numbered like [41]. You choose clips ONLY by sentence "
    "numbers (never timestamps). A clip is a continuous range start_sentence..end_sentence, "
    "roughly 20-90 seconds. It must start with a strong hook, be understandable without any "
    "other context (don't start with 'and', 'but', 'so', 'as I said'), and end on a punchline, "
    "conclusion or natural pause. Score each 0-10: hook, emotion, self_contained, punchline, "
    "ending_complete. hook_sentence = the single strongest sentence inside the range; "
    "use_teaser = true only if that sentence is NOT the first one. Write title, description and "
    "hashtags in the same language as the transcript. Be strict: return fewer clips rather than "
    "weak ones. Return JSON only."
)


# ---------------------------------------------------------------- LLM backends
class LlamaCppBackend:
    def __init__(self, cfg: dict):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ClipperError("llama-cpp-python not installed (build with CUDA), "
                               "or set llm.backend: ollama") from e
        from pathlib import Path
        if not Path(cfg["model_path"]).exists():
            raise ClipperError(f"LLM model not found: {cfg['model_path']}. "
                               "Download a GGUF and set llm.model_path in config.yaml.")
        self.llm = Llama(model_path=cfg["model_path"], n_ctx=cfg["n_ctx"],
                         n_gpu_layers=cfg["n_gpu_layers"], verbose=False)

    def json(self, system, user, schema, max_tokens=1500) -> dict:
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.2, max_tokens=max_tokens)
        return json.loads(r["choices"][0]["message"]["content"])

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
        self.llm = None


class OllamaBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = cfg["ollama_url"].rstrip("/")

    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception as e:
            raise ClipperError(f"Ollama not reachable at {self.url}: {e}") from e

    def json(self, system, user, schema, max_tokens=1500) -> dict:
        r = self._post("/api/chat", {
            "model": self.cfg["ollama_model"], "stream": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.2, "num_ctx": self.cfg["n_ctx"], "num_predict": max_tokens},
        })
        return json.loads(r["message"]["content"])

    def close(self):
        try:
            self._post("/api/generate", {"model": self.cfg["ollama_model"], "keep_alive": 0})
        except Exception:
            pass


class LazyLLM:
    """Loads the LLM on first use (cache hits never touch the GPU); unloads on close()."""

    def __init__(self, cfg: dict):
        self.cfg, self.backend = cfg, None

    def _ensure(self):
        if self.backend is None:
            GPU.acquire("llm")
            try:
                cls = OllamaBackend if self.cfg["backend"] == "ollama" else LlamaCppBackend
                self.backend = cls(self.cfg)
            except Exception:
                GPU.release()
                raise

    def json(self, system, user, schema, max_tokens=1500, retries=2) -> dict:
        self._ensure()
        err = None
        for _ in range(retries + 1):
            try:
                return self.backend.json(system, user, schema, max_tokens)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                err = e
        raise ClipperError(f"LLM returned invalid JSON after retries: {err}")

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
            GPU.release()


# ---------------------------------------------------------------- window scoring
def prefilter_windows(windows, sentences, signals, weights, max_windows):
    if len(windows) <= max_windows:
        return windows
    curve = interest_curve(signals, weights)
    scored = []
    for w in windows:
        s, e = sentences[w[0]]["start"], sentences[w[-1]]["end"]
        scored.append((span_mean(curve, s, e), w))
    top = sorted(scored, key=lambda x: -x[0])[:max_windows]
    return sorted((w for _, w in top), key=lambda w: w[0])


def score_window(llm, sentences, window, per_window) -> list[dict]:
    lines = [f"[{sentences[i]['id']}] {sentences[i]['start']:.1f}-{sentences[i]['end']:.1f} "
             f"{sentences[i]['text']}" for i in window]
    user = f"Pick up to {per_window} best clips from this transcript window.\n\n" + "\n".join(lines)
    res = llm.json(SYSTEM, user, CAND_SCHEMA, max_tokens=1800)
    lo, hi = sentences[window[0]]["id"], sentences[window[-1]]["id"]
    out = []
    for c in res.get("candidates", []):
        try:
            s, e = int(c["start_sentence"]), int(c["end_sentence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (lo <= s <= e <= hi):
            continue
        scores = {k: max(0, min(10, int(c.get("scores", {}).get(k, 0)))) for k in SCORE_KEYS}
        out.append({
            "start_sentence": s, "end_sentence": e, "scores": scores,
            "hook_sentence": int(c.get("hook_sentence", s)),
            "use_teaser": bool(c.get("use_teaser", False)),
            "title": str(c.get("title", "")).strip(),
            "description": str(c.get("description", "")).strip(),
            "hashtags": [str(h) for h in c.get("hashtags", [])][:8],
            "reason": str(c.get("reason", "")).strip(),
        })
    return out


# ---------------------------------------------------------------- deterministic selection
BAD_STARTS = re.compile(r"^(dan|tapi|terus|lalu|makanya|jadi|nah|and|but|so|then|because|which)\b",
                        re.IGNORECASE)
TERMINAL = re.compile(r"[.!?…][\"')\]]*$")


def _dur(sentences, s_id, e_id):
    return sentences[e_id - 1]["end"] - sentences[s_id - 1]["start"]


def fit_duration(sentences, s_id, e_id, min_s, max_s):
    """Trim/extend whole sentences so the clip fits [min_s, max_s]. Returns (s,e) or None."""
    n = len(sentences)
    while _dur(sentences, s_id, e_id) > max_s and e_id > s_id:
        e_id -= 1
    while _dur(sentences, s_id, e_id) < min_s and e_id < n:
        e_id += 1
        if _dur(sentences, s_id, e_id) > max_s:
            return None
    return (s_id, e_id) if min_s <= _dur(sentences, s_id, e_id) <= max_s else None


def adjust_ending(sentences, s_id, e_id, min_s, max_s, max_adjust):
    """Pick the best end sentence within +-max_adjust: terminal punctuation + natural pause."""
    best, best_score = e_id, -1e9
    for cand in range(e_id - max_adjust, e_id + max_adjust + 1):
        if cand < s_id or cand > len(sentences):
            continue
        if not (min_s <= _dur(sentences, s_id, cand) <= max_s):
            continue
        sc = 0.0
        sc += 1.0 if TERMINAL.search(sentences[cand - 1]["text"].strip()) else 0.0
        nxt = sentences[cand]["start"] - sentences[cand - 1]["end"] if cand < len(sentences) else 1.0
        sc += 1.0 if nxt > 0.5 else 0.0
        sc -= 0.3 * abs(cand - e_id)
        if sc > best_score:
            best, best_score = cand, sc
    return best


def adjust_start(sentences, s_id, e_id, min_s):
    """Skip up to 2 leading sentences that begin with a dangling connector."""
    for _ in range(2):
        if BAD_STARTS.match(sentences[s_id - 1]["text"].strip()) and s_id < e_id \
                and _dur(sentences, s_id + 1, e_id) >= min_s:
            s_id += 1
        else:
            break
    return s_id


def llm_score(scores: dict) -> float:
    return sum(scores.get(k, 0) for k in SCORE_KEYS) / (10.0 * len(SCORE_KEYS))


def final_score(llm_s: float, sig: dict, weights: dict) -> float:
    parts = {"llm": llm_s, **sig}
    num = den = 0.0
    for k, v in parts.items():
        w = float(weights.get(k, 0))
        if v is None or w <= 0:
            continue
        num += w * v
        den += w
    return num / den if den else 0.0


def overlaps(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def nms(cands: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        if not any(overlaps(c, k) for k in kept):
            kept.append(c)
    return kept


_WORD = re.compile(r"\w+", re.UNICODE)


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(_WORD.findall(a.lower())), Counter(_WORD.findall(b.lower()))
    dot = sum(ca[k] * cb[k] for k in ca)
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def build_teaser(c, sentences, max_seconds):
    h = c.get("hook_sentence")
    if not c.get("use_teaser") or not h or not (c["start_sentence"] < h <= c["end_sentence"]):
        return None
    t0, t1 = sentences[h - 1]["start"], sentences[h - 1]["end"]
    i = h
    while t1 - t0 < 2.0 and i < c["end_sentence"]:
        i += 1
        t1 = sentences[i - 1]["end"]
    t1 = min(t1, t0 + max_seconds)
    return {"start": t0, "end": t1, "sentence": h} if t1 - t0 >= 1.5 else None


def finalize(raw: list[dict], sentences, signals, sel: dict, teaser_cfg: dict,
             llm=None) -> list[dict]:
    """Cheap, deterministic stage. Re-run freely when num_clips/duration/weights change."""
    min_s, max_s = sel["min_seconds"], sel["max_seconds"]
    weights = sel["weights"]
    cands = []
    for c in raw:
        s = adjust_start(sentences, c["start_sentence"], c["end_sentence"], min_s)
        fit = fit_duration(sentences, s, c["end_sentence"], min_s, max_s)
        if not fit:
            continue
        s, e = fit
        e = adjust_ending(sentences, s, e, min_s, max_s, sel["ending_max_adjust_sentences"])
        start, end = sentences[s - 1]["start"], sentences[e - 1]["end"]
        sig = {k: span_mean(signals.get(k), start, end, GRID)
               for k in ("audio_energy", "laughter", "speech_rate")}
        sig["heatmap"] = span_mean(signals["heatmap"], start, end, GRID) if signals.get("heatmap") else None
        ls = llm_score(c["scores"])
        item = dict(c, start_sentence=s, end_sentence=e, start=start, end=end,
                    duration=round(end - start, 2), llm_score=round(ls, 3),
                    signals={k: (round(v, 3) if v is not None else None) for k, v in sig.items()},
                    score=round(final_score(ls, sig, weights), 4),
                    text=" ".join(x["text"] for x in sentences[s - 1:e]))
        item["teaser"] = build_teaser(item, sentences, teaser_cfg["max_seconds"]) \
            if teaser_cfg["enabled"] else None
        cands.append(item)

    picked: list[dict] = []
    for c in nms(cands):
        if any(cosine(c["text"], p["text"]) >= sel["dedup_threshold"] for p in picked):
            continue
        if sel.get("self_check") and llm is not None and not self_check(llm, c["text"]):
            continue
        picked.append(c)
        if len(picked) >= sel["num_clips"]:
            break
    picked.sort(key=lambda x: -x["score"])
    for i, c in enumerate(picked, 1):
        c["rank"] = i
    return picked


def self_check(llm, text: str) -> bool:
    schema = {"type": "object", "properties": {"standalone": {"type": "boolean"}},
              "required": ["standalone"]}
    try:
        r = llm.json("Answer JSON only. Can this clip be fully understood by someone who has not "
                     "seen the rest of the video?", text, schema, max_tokens=30)
        return bool(r.get("standalone", True))
    except Exception:
        return True
