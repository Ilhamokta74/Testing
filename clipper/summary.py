from __future__ import annotations

import csv
from pathlib import Path

from .util import fmt_ts


def write_summary(out_dir: Path, meta: dict, clips: list[dict], formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "md" in formats:
        lines = [f"# {meta.get('title', meta['name'])}", "",
                 f"Source: {meta.get('source_url') or meta['video']}", ""]
        for c in clips:
            lines += [
                f"## Clip {c['rank']:02d}: {c['title']}", "",
                f"- File: `{c.get('file', 'not rendered')}`",
                f"- Source range: {fmt_ts(c['start'])} - {fmt_ts(c['end'])} ({c['duration']:.0f}s)",
                f"- Score: {c['score']:.2f} (LLM {c['llm_score']:.2f})",
                f"- Reason: {c['reason']}", "",
                c["description"], "", " ".join(c["hashtags"]), "",
            ]
        (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if "csv" in formats:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["no", "file", "start", "end", "duration", "score", "title",
                        "description", "hashtags", "reason"])
            for c in clips:
                w.writerow([c["rank"], c.get("file", ""), round(c["start"], 2), round(c["end"], 2),
                            c["duration"], c["score"], c["title"], c["description"],
                            " ".join(c["hashtags"]), c["reason"]])
