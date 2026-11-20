from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import audio, highlight, ingest, refine, segment, summary, transcribe
from .cache import StageCache
from .render import render_clip
from .util import ClipperError, hash_key, read_json, write_json

Progress = Callable[[float, str], None]
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunOptions:
    source: str = ""
    is_url: bool = False
    dry_run: bool = False
    no_cache: bool = False
    fresh: bool = False
    no_teaser: bool = False
    clips: str | None = None          # "1,3,5" subset for rendering


def _p(progress, frac, msg):
    if progress:
        progress(frac, msg)


def _glossary(cfg: dict, override: str | None = None) -> list[str]:
    return refine.load_glossary(override or cfg["transcript_fix"]["glossary_path"])


def analyze(cfg: dict, opts: RunOptions, progress: Progress | None = None):
    """Ingest -> transcript -> sentences -> signals -> candidates. Returns (work, meta, ctx)."""
    cfg = copy.deepcopy(cfg)
    if opts.no_teaser:
        cfg["teaser"]["enabled"] = False
    work_root = Path(cfg["cache"]["dir"])

    _p(progress, 0.02, "Ingest")
    work, meta = ingest.ingest(opts.source, opts.is_url, work_root, fresh=opts.fresh)
    cache = StageCache(work, enabled=True, fresh=opts.fresh)
    glossary = _glossary(cfg)

    wav = work / "audio.wav"
    if opts.fresh or not wav.exists():
        _p(progress, 0.05, "Extracting audio")
        audio.extract_wav(Path(meta["video"]), wav)

    t_key = hash_key(meta["hash"], cfg["whisper"], cfg["language"], glossary)
    tr = cache.get("transcript", t_key)
    if tr is None:
        _p(progress, 0.08, "Transcribing (Whisper)")
        tr = transcribe.transcribe(wav, cfg["whisper"]["model"], cfg["whisper"]["compute_type"],
                                   cfg["language"], glossary,
                                   lambda f, m: _p(progress, 0.08 + 0.4 * f, m),
                                   device=cfg["whisper"].get("device", "auto"))
        cache.put("transcript", t_key, tr)
    if not tr["words"]:
        raise ClipperError("Transcript is empty (no speech detected).")

    words = refine.fuzzy_fix(tr["words"], glossary)
    sentences = segment.build_sentences(words)
    write_json(work / "sentences.json", sentences)

    _p(progress, 0.5, "Audio signals")
    s_key = hash_key(t_key, "signals-v1")
    signals = cache.get("signals", s_key)
    if signals is None:
        signals = audio.compute_signals(wav, words, meta["duration"], meta.get("heatmap"))
        cache.put("signals", s_key, signals)

    sel = cfg["selection"]
    windows = segment.make_windows(sentences, sel["window_minutes"] * 60, sel["window_overlap_seconds"])
    windows = highlight.prefilter_windows(windows, sentences, signals, sel["weights"], sel["max_windows"])

    llm = highlight.LazyLLM(cfg["llm"])
    try:
        r_key = hash_key(t_key, cfg["llm"]["model_path"], cfg["llm"]["ollama_model"],
                         cfg["llm"]["backend"], sel["window_minutes"], sel["window_overlap_seconds"],
                         sel["max_windows"], sel["per_window"], sel["weights"])
        raw = cache.get("candidates_raw", r_key)
        if raw is None:
            raw = []
            for i, w in enumerate(windows):
                _p(progress, 0.55 + 0.3 * i / max(1, len(windows)), f"LLM scoring window {i + 1}/{len(windows)}")
                raw += highlight.score_window(llm, sentences, w, sel["per_window"])
            cache.put("candidates_raw", r_key, raw)

        _p(progress, 0.87, "Selecting clips")
        cands = highlight.finalize(raw, sentences, signals, sel, cfg["teaser"],
                                   llm if sel["self_check"] else None)
        if not cands:
            raise ClipperError("No clips passed the duration/quality filters. "
                               "Try a wider --min/--max range.")

        if cfg["transcript_fix"]["llm_correction"]:
            _p(progress, 0.9, "Correcting transcript")
            for c in cands:
                i0, i1 = sentences[c["start_sentence"] - 1]["w0"], sentences[c["end_sentence"] - 1]["w1"]
                fixed = refine.llm_correct(llm, words[i0:i1 + 1], glossary)
                words[i0:i1 + 1] = fixed
    finally:
        llm.close()

    write_json(work / "candidates.json", cands)
    write_json(work / "words.json", words)
    return work, meta, {"cands": cands, "words": words, "cfg": cfg}


def render_all(work: Path, meta: dict, cands: list[dict], words: list[dict], cfg: dict,
               only: list[int] | None, progress: Progress | None = None, out_dir: Path | None = None):
    name = meta["name"]
    out = out_dir or Path(cfg["output"]["dir"]) / name
    chosen = [c for c in cands if not only or c["rank"] in only]
    if not chosen:
        raise ClipperError("No matching clips to render.")
    fonts = ROOT / "assets" / "fonts"
    for i, c in enumerate(chosen):
        _p(progress, 0.92 + 0.08 * i / len(chosen), f"Rendering clip {c['rank']:02d}")
        f = out / f"clip_{c['rank']:02d}.mp4"
        render_clip(meta, words, c, f, cfg, fonts)
        c["file"] = f.name
        write_json(out / f"clip_{c['rank']:02d}.json", c)
    summary.write_summary(out, meta, cands, cfg["output"]["summary"])
    return out


def run(cfg: dict, opts: RunOptions, progress: Progress | None = None):
    work, meta, ctx = analyze(cfg, opts, progress)
    cands, words, cfg2 = ctx["cands"], ctx["words"], ctx["cfg"]
    out = Path(cfg2["output"]["dir"]) / meta["name"]
    if opts.dry_run:
        summary.write_summary(out, meta, cands, cfg2["output"]["summary"])
        return work, meta, cands, out, False
    only = [int(x) for x in opts.clips.split(",")] if opts.clips else None
    render_all(work, meta, cands, words, cfg2, only, progress)
    _p(progress, 1.0, "Done")
    if opts.no_cache:
        StageCache(work).cleanup()
    return work, meta, cands, out, True


def render_from_work(cfg: dict, work: Path, only: list[int] | None, style: str | None,
                     progress: Progress | None = None):
    meta = ingest.load_meta(work)
    cands_p, words_p = work / "candidates.json", work / "words.json"
    if not cands_p.exists() or not words_p.exists():
        raise ClipperError("candidates.json / words.json missing. Run `clipper run --dry-run` first.")
    cfg = copy.deepcopy(cfg)
    if style:
        cfg["captions"]["preset"] = style
    out = render_all(work, meta, read_json(cands_p), read_json(words_p), cfg, only, progress)
    return meta, out
