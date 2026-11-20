from __future__ import annotations

import re

PRESETS = {
    "tiktok": {"uppercase": True, "scale": 112, "outline": 7, "shadow": 2,
               "primary": "#FFFFFF", "highlight": "#FFD400"},
    "clean": {"uppercase": False, "scale": 100, "outline": 5, "shadow": 1,
              "primary": "#FFFFFF", "highlight": "#4FD1FF"},
}


def _ass_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&".upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def map_words(words: list[dict], segments: list[tuple[float, float]]) -> list[dict]:
    """Map source-time words onto output time for a list of (start, end) source segments."""
    out, offset = [], 0.0
    for s, e in segments:
        for w in words:
            if w["start"] >= s - 0.01 and w["end"] <= e + 0.05:
                out.append(dict(w, start=offset + max(0.0, w["start"] - s),
                                end=offset + min(e, w["end"]) - s))
        offset += e - s
    return out


def build_ass(words: list[dict], cap_cfg: dict, resolution=(1080, 1920)) -> str:
    preset = dict(PRESETS.get(cap_cfg.get("preset", "tiktok"), PRESETS["tiktok"]))
    if cap_cfg.get("highlight_color") and cap_cfg.get("preset", "tiktok") == "tiktok":
        preset["highlight"] = cap_cfg["highlight_color"]
    W, H = resolution
    margin_v = int(H * cap_cfg.get("margin_v_pct", 0.22))
    per = max(1, int(cap_cfg.get("words_per_line", 3)))
    hi, base = _ass_color(preset["highlight"]), _ass_color(preset["primary"])

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{cap_cfg.get('font', 'Arial')},{cap_cfg.get('font_size', 78)},{base},{base},"
        f"&H00000000&,&H64000000&,-1,0,0,0,100,100,0,0,1,{preset['outline']},{preset['shadow']},2,"
        f"60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (len(cur) >= per or w["start"] - cur[-1]["end"] > 0.8):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?…]$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    def fmt(text: str) -> str:
        text = text.replace("{", "(").replace("}", ")").replace("\\", "")
        return text.upper() if preset["uppercase"] else text

    lines = []
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for i, w in enumerate(g):
            start = w["start"]
            end = g[i + 1]["start"] if i + 1 < len(g) else w["end"] + 0.15
            if i + 1 == len(g) and next_start is not None:
                end = min(end, next_start)
            end = max(end, start + 0.05)
            parts = []
            for j, x in enumerate(g):
                if j == i:
                    parts.append(f"{{\\c{hi}\\fscx{preset['scale']}\\fscy{preset['scale']}}}"
                                 f"{fmt(x['w'])}{{\\r}}")
                else:
                    parts.append(fmt(x["w"]))
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"
