from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def load_glossary(path: str | Path | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def _ratio(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b)
    except ImportError:
        return SequenceMatcher(None, a, b).ratio() * 100


def _split(token: str) -> tuple[str, str, str]:
    m = PUNCT.match(token)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")


def fuzzy_fix(words: list[dict], glossary: list[str], threshold: float = 82.0) -> list[dict]:
    """Replace near-miss spellings with glossary terms. Keeps word count and timestamps."""
    if not glossary:
        return words
    out = [dict(w) for w in words]
    entries = [(g, g.split()) for g in glossary]
    for term, toks in entries:
        k = len(toks)
        target = " ".join(toks).lower()
        for i in range(len(out) - k + 1):
            core = [_split(out[i + j]["w"]) for j in range(k)]
            cand = " ".join(c[1] for c in core).lower()
            if not cand or cand == target:
                continue
            if abs(len(cand) - len(target)) > 3:
                continue
            if _ratio(cand, target) >= threshold:
                for j in range(k):
                    pre, _, post = core[j]
                    out[i + j]["w"] = f"{pre}{toks[j]}{post}"
    return out


def llm_correct(llm, words: list[dict], glossary: list[str]) -> list[dict]:
    """Ask the LLM to fix spelling of a clip's words. Applied only when the token count
    is unchanged, so per-word timestamps stay valid."""
    if not words:
        return words
    original = [w["w"] for w in words]
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    system = ("You fix speech-to-text spelling errors (names, jargon, Indonesian/English mix). "
              "Do NOT add, remove, merge, split or reorder words. Keep punctuation. "
              "Return JSON {\"text\": \"...\"} with exactly the same number of words.")
    user = ""
    if glossary:
        user += "Known terms: " + ", ".join(glossary) + "\n"
    user += "Text:\n" + " ".join(original)
    try:
        res = llm.json(system, user, schema, max_tokens=len(original) * 4 + 64)
        fixed = str(res.get("text", "")).split()
    except Exception:
        return words
    if len(fixed) != len(original):
        return words
    out = [dict(w) for w in words]
    for w, f in zip(out, fixed):
        w["w"] = f
    return out
