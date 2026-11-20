from __future__ import annotations

import copy
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import pipeline
from .config import load_config
from .util import ClipperError, fmt_ts

app = typer.Typer(add_completion=False, help="Local video clipper: long video -> 9:16 clips.")
console = Console()


def _cfg(config: Optional[Path], lang, clips, min_s, max_s, glossary, style):
    cfg = load_config(config)
    if lang:
        cfg["language"] = lang
    if clips:
        cfg["selection"]["num_clips"] = clips
    if min_s:
        cfg["selection"]["min_seconds"] = min_s
    if max_s:
        cfg["selection"]["max_seconds"] = max_s
    if glossary:
        cfg["transcript_fix"]["glossary_path"] = str(glossary)
    if style:
        cfg["captions"]["preset"] = style
    return cfg


def _table(cands):
    t = Table(show_lines=False)
    for col in ("#", "Time", "Dur", "Score", "Title", "Reason"):
        t.add_column(col, overflow="fold")
    for c in cands:
        t.add_row(str(c["rank"]), f"{fmt_ts(c['start'])}-{fmt_ts(c['end'])}", f"{c['duration']:.0f}s",
                  f"{c['score']:.2f}", c["title"], c["reason"])
    console.print(t)


@app.command()
def run(
    url: Optional[str] = typer.Option(None, "--url", help="YouTube/URL supported by yt-dlp"),
    file: Optional[Path] = typer.Option(None, "--file", help="Local video file"),
    lang: Optional[str] = typer.Option(None, "--lang", help="auto | id | en"),
    clips: Optional[int] = typer.Option(None, "--clips", help="Number of clips"),
    min_s: Optional[int] = typer.Option(None, "--min", help="Min clip seconds"),
    max_s: Optional[int] = typer.Option(None, "--max", help="Max clip seconds"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stop before rendering"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Delete work folder when done"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore old cache"),
    no_teaser: bool = typer.Option(False, "--no-teaser"),
    glossary: Optional[Path] = typer.Option(None, "--glossary"),
    style: Optional[str] = typer.Option(None, "--style", help="tiktok | clean"),
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Full pipeline: ingest -> transcribe -> select -> render."""
    if bool(url) == bool(file):
        raise typer.BadParameter("Provide exactly one of --url or --file.")
    cfg = _cfg(config, lang, clips, min_s, max_s, glossary, style)
    opts = pipeline.RunOptions(source=url or str(file), is_url=bool(url), dry_run=dry_run,
                               no_cache=no_cache, fresh=fresh, no_teaser=no_teaser)
    try:
        with Progress(TextColumn("[bold]{task.description}"), BarColumn(),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Starting", total=100)

            def cb(frac, msg):
                prog.update(task, completed=int(frac * 100), description=msg)

            work, meta, cands, out, rendered = pipeline.run(cfg, opts, cb)
    except ClipperError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    _table(cands)
    if rendered:
        console.print(f"[green]Done.[/green] Output: {out}")
    else:
        console.print(f"Dry run complete. Pick clips and run:\n  clipper render --work {work} --clips 1,3,5")


@app.command()
def render(
    work: Path = typer.Option(..., "--work", help="work/<video name> folder"),
    clips: Optional[str] = typer.Option(None, "--clips", help="e.g. 1,3,5 (default: all)"),
    style: Optional[str] = typer.Option(None, "--style"),
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Render selected clips from cache (no Whisper/LLM re-run)."""
    cfg = load_config(config)
    only = [int(x) for x in clips.split(",")] if clips else None
    try:
        with Progress(TextColumn("[bold]{task.description}"), BarColumn(), console=console) as prog:
            task = prog.add_task("Rendering", total=100)
            _, out = pipeline.render_from_work(
                cfg, work, only, style, lambda f, m: prog.update(task, completed=int(f * 100), description=m))
    except ClipperError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Done.[/green] Output: {out}")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    console.print(f"[{'green' if ok else 'red'}]{'OK ' if ok else 'FAIL'}[/] {name} {detail}")
    return ok


@app.command()
def doctor(config: Optional[Path] = typer.Option(None, "--config")):
    """Check environment: ffmpeg/NVENC/libass, CUDA/VRAM, models, fonts."""
    cfg = load_config(config)
    ok = True
    console.print(f"Python {platform.python_version()} on {platform.system()}")
    ok &= _check("Python >= 3.10", sys.version_info >= (3, 10))
    ff = shutil.which("ffmpeg")
    ok &= _check("ffmpeg", bool(ff), ff or "not in PATH")
    ok &= _check("ffprobe", bool(shutil.which("ffprobe")))
    if ff:
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        flt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        _check("NVENC (NVIDIA only)", "h264_nvenc" in enc, "(optional; AMD users: h264_amf or libx264)")
        _check("AMF (AMD encoder)", "h264_amf" in enc, "(optional, Windows)")
        ok &= _check("libass ('ass' filter)", " ass " in flt)
    for mod, pipname in [("yt_dlp", "yt-dlp"), ("faster_whisper", "faster-whisper"),
                         ("numpy", "numpy"), ("yaml", "pyyaml"), ("typer", "typer")]:
        try:
            __import__(mod)
            _check(pipname, True)
        except ImportError:
            ok &= _check(pipname, False, f"pip install {pipname}")
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            _check("CUDA", True, f"{p.name}, {p.total_memory / 2**30:.1f} GB VRAM")
        else:
            _check("CUDA", True, "no NVIDIA GPU: CPU mode (set whisper.device: cpu)")
    except ImportError:
        smi = shutil.which("nvidia-smi")
        _check("NVIDIA driver (optional)", True, "found" if smi else "none: CPU mode")
    llm = cfg["llm"]
    if llm["backend"] == "llama_cpp":
        try:
            import llama_cpp  # noqa: F401
            _check("llama-cpp-python", True)
        except ImportError:
            ok &= _check("llama-cpp-python", False, "or set llm.backend: ollama")
        ok &= _check("LLM model file", Path(llm["model_path"]).exists(), llm["model_path"])
    else:
        import urllib.request
        try:
            urllib.request.urlopen(llm["ollama_url"], timeout=3)
            _check("Ollama", True, llm["ollama_url"])
        except Exception:
            ok &= _check("Ollama", False, f"not reachable at {llm['ollama_url']}")
    fonts = pipeline.ROOT / "assets" / "fonts"
    _check("Font in assets/fonts", any(fonts.glob("*.[to]tf")),
           "(put e.g. Montserrat-ExtraBold.ttf here; else system fonts are used)")
    raise typer.Exit(0 if ok else 1)


@app.command("download-models")
def download_models(config: Optional[Path] = typer.Option(None, "--config")):
    """Download the Whisper model once. After this everything runs offline."""
    cfg = load_config(config)
    from faster_whisper import WhisperModel
    console.print(f"Downloading Whisper '{cfg['whisper']['model']}' ...")
    WhisperModel(cfg["whisper"]["model"], device="cpu", compute_type="int8")
    console.print("[green]Whisper ready.[/green]")
    console.print("LLM: download a GGUF (e.g. Qwen 7B Instruct Q4_K_M) into models/ and set "
                  "llm.model_path, or `ollama pull <model>` and set llm.backend: ollama.")


if __name__ == "__main__":
    app()
