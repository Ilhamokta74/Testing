from __future__ import annotations

import json
import shutil
from pathlib import Path

from .util import ClipperError, file_hash, run, slugify, write_json, read_json


def probe(video: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", str(video)]).stdout
    data = json.loads(out)
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not vs:
        raise ClipperError(f"No video stream found in {video}")
    num, _, den = (vs.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float(data.get("format", {}).get("duration") or vs.get("duration") or 0)
    return {"width": int(vs["width"]), "height": int(vs["height"]),
            "fps": fps, "duration": duration}


def _download(url: str, work_root: Path, fresh: bool) -> tuple[Path, dict]:
    try:
        import yt_dlp
    except ImportError as e:
        raise ClipperError("yt-dlp is not installed: pip install yt-dlp") from e

    base_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ClipperError(_explain_dl_error(str(e))) from e
    if info.get("_type") == "playlist":
        raise ClipperError("URL is a playlist. Pass a single video URL instead.")
    if info.get("is_live"):
        raise ClipperError("Live streams are not supported. Wait until the VOD is available.")

    name = slugify(info.get("title") or info.get("id") or "video")
    work = work_root / name
    work.mkdir(parents=True, exist_ok=True)
    existing = next(iter(sorted(work.glob("source.*"))), None)
    if existing and not fresh and existing.suffix in (".mp4", ".mkv", ".webm"):
        return existing, info

    opts = dict(base_opts, outtmpl=str(work / "source.%(ext)s"),
                format="bv*[height<=1080]+ba/b[height<=1080]/b",
                merge_output_format="mp4")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise ClipperError(_explain_dl_error(str(e))) from e
    video = next(iter(sorted(work.glob("source.*"))), None)
    if not video:
        raise ClipperError("Download finished but no file was produced.")
    return video, info


def _explain_dl_error(msg: str) -> str:
    low = msg.lower()
    if "private" in low:
        return "Video is private."
    if "geo" in low or "not available in your country" in low:
        return "Video is geo-blocked in your region."
    if "sign in" in low or "age" in low:
        return "Video requires login/age verification (try yt-dlp --cookies-from-browser)."
    return f"Could not download video: {msg.strip()[-300:]}"


def ingest(source: str, is_url: bool, work_root: Path, fresh: bool = False) -> tuple[Path, dict]:
    """Returns (work_dir, meta). meta is also written to work_dir/meta.json."""
    info: dict = {}
    if is_url:
        video, info = _download(source, work_root, fresh)
        work = video.parent
    else:
        src = Path(source).expanduser()
        if not src.is_file():
            raise ClipperError(f"File not found: {src}")
        work = work_root / slugify(src.stem)
        work.mkdir(parents=True, exist_ok=True)
        video = work / f"source{src.suffix.lower()}"
        if fresh or not video.exists():
            try:
                video.unlink(missing_ok=True)
                video.hardlink_to(src)
            except Exception:
                shutil.copy2(src, video)

    meta = probe(video)
    meta.update({
        "name": work.name, "video": str(video), "hash": file_hash(video),
        "title": info.get("title") or work.name, "source_url": source if is_url else None,
        "heatmap": info.get("heatmap"), "chapters": info.get("chapters"),
        "uploader": info.get("uploader"),
    })
    write_json(work / "meta.json", meta)
    return work, meta


def load_meta(work: Path) -> dict:
    p = work / "meta.json"
    if not p.exists():
        raise ClipperError(f"{p} not found. Run `clipper run` first.")
    return read_json(p)
