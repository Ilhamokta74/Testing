from __future__ import annotations

import logging

log = logging.getLogger("clipper")


def plan_layout(src_w: int, src_h: int, mode: str) -> dict:
    """Phase 1: center crop (or blurred-background fit). face/speaker arrive in Phase 2."""
    if mode in ("face", "speaker"):
        log.warning("reframe.mode=%s is Phase 2; using center crop.", mode)
        mode = "center"
    if src_w / src_h <= 9 / 16 + 0.01:      # already vertical
        return {"layout": "scale"}
    if mode == "blur":
        return {"layout": "blur"}
    w = int(src_h * 9 / 16) // 2 * 2
    x = (src_w - w) // 2
    return {"layout": "crop", "x": x, "y": 0, "w": w, "h": src_h}
