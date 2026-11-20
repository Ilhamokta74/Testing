from __future__ import annotations

import subprocess
from pathlib import Path

from .captions import build_ass, map_words
from .reframe import plan_layout
from .util import ClipperError, run


def _esc(path: Path) -> str:
    s = str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return s


def clip_segments(cand: dict, end_padding: float) -> list[tuple[float, float]]:
    segs = []
    if cand.get("teaser"):
        segs.append((cand["teaser"]["start"], cand["teaser"]["end"]))
    segs.append((max(0.0, cand["start"] - 0.05), cand["end"] + end_padding))
    return segs


def encoder_args(name: str, cq: int) -> list[str]:
    if name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(cq), "-b:v", "0",
                "-pix_fmt", "yuv420p"]
    if name == "h264_amf":   # AMD hardware encoder (Windows)
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                "-qp_i", str(cq), "-qp_p", str(cq), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq), "-pix_fmt", "yuv420p"]


def build_filter(segs, base, layout, res, ass_name, fonts_dir, loudnorm) -> str:
    W, H = res
    parts = []
    for i, (a, b) in enumerate(segs):
        parts.append(f"[0:v]trim=start={a - base:.3f}:end={b - base:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a - base:.3f}:end={b - base:.3f},asetpts=PTS-STARTPTS[a{i}]")
    cat_in = "".join(f"[v{i}][a{i}]" for i in range(len(segs)))
    parts.append(f"{cat_in}concat=n={len(segs)}:v=1:a=1[cv][ca]")

    ass = f"ass='{_esc(Path(ass_name))}'" + (f":fontsdir='{_esc(fonts_dir)}'" if fonts_dir else "")
    kind = layout["layout"]
    if kind == "crop":
        parts.append(f"[cv]crop={layout['w']}:{layout['h']}:{layout['x']}:{layout['y']},"
                     f"scale={W}:{H},setsar=1,{ass}[v]")
    elif kind == "blur":
        parts.append("[cv]split[bg][fg]")
        parts.append(f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                     "boxblur=30:5[b]")
        parts.append(f"[fg]scale={W}:-2,setsar=1[f]")
        parts.append(f"[b][f]overlay=(W-w)/2:(H-h)/2,{ass}[v]")
    else:
        parts.append(f"[cv]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,{ass}[v]")
    parts.append("[ca]loudnorm=I=-16:TP=-1.5:LRA=11[a]" if loudnorm else "[ca]anull[a]")
    return ";".join(parts)


def render_clip(meta: dict, words: list[dict], cand: dict, out_path: Path,
                cfg: dict, fonts_dir: Path | None = None) -> Path:
    rcfg = cfg["render"]
    res = tuple(rcfg["resolution"])
    segs = clip_segments(cand, rcfg["end_padding"])
    base = min(a for a, _ in segs)
    span = max(b for _, b in segs) - base

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = out_path.with_suffix(".ass")
    ass_path.write_text(build_ass(map_words(words, segs), cfg["captions"], res), encoding="utf-8")

    layout = plan_layout(meta["width"], meta["height"], cfg["reframe"]["mode"])
    fonts = fonts_dir if fonts_dir and Path(fonts_dir).exists() else None

    encoders = [rcfg["encoder"]] + (["libx264"] if rcfg["encoder"] != "libx264" else [])
    last: Exception | None = None
    for enc in encoders:
        fc = build_filter(segs, base, layout, res, ass_path.name, fonts, rcfg["loudnorm"])
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{base:.3f}", "-t", f"{span:.3f}", "-i", meta["video"],
               "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
               *encoder_args(enc, rcfg["cq"]), "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", out_path.name]
        try:
            run(cmd, cwd=out_path.parent)   # cwd = output dir keeps the .ass path simple
            return out_path
        except ClipperError as e:
            last = e
    raise ClipperError(f"Render failed: {last}")
