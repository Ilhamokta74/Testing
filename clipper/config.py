from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "language": "auto",
    "whisper": {"model": "large-v3-turbo", "compute_type": "int8_float16", "device": "auto"},
    "llm": {
        "backend": "llama_cpp", "model_path": "models/qwen-7b-instruct-q4_k_m.gguf",
        "ollama_model": "qwen2.5:7b-instruct", "ollama_url": "http://localhost:11434",
        "n_ctx": 8192, "n_gpu_layers": -1,
    },
    "selection": {
        "num_clips": 8, "min_seconds": 25, "max_seconds": 75, "window_minutes": 4,
        "window_overlap_seconds": 45, "max_windows": 20, "per_window": 3,
        "self_check": False, "dedup_threshold": 0.75, "ending_max_adjust_sentences": 3,
        "weights": {"llm": 0.55, "audio_energy": 0.15, "laughter": 0.10,
                    "speech_rate": 0.05, "heatmap": 0.15},
    },
    "teaser": {"enabled": True, "max_seconds": 3},
    "transcript_fix": {"glossary_path": "glossary.txt", "llm_correction": True},
    "cache": {"dir": "work", "keep": True},
    "output": {"dir": "output", "summary": ["md", "csv"]},
    "reframe": {"mode": "center", "smoothing": 0.15, "dead_zone": 0.12, "fallback": "blur"},
    "captions": {"preset": "tiktok", "words_per_line": 3, "highlight_color": "#FFD400",
                 "font": "Montserrat ExtraBold", "font_size": 78, "margin_v_pct": 0.22},
    "render": {"encoder": "h264_nvenc", "resolution": [1080, 1920], "cq": 19,
               "loudnorm": True, "end_padding": 0.4},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path("config.yaml")
    if p.exists():
        return deep_merge(DEFAULTS, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return copy.deepcopy(DEFAULTS)
