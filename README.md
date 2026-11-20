# Clipper (Phase 1 MVP)

Local, offline, free video clipper: long video -> vertical 9:16 clips with per-word captions.
Built for RTX 3060 6 GB: only one GPU model (Whisper or LLM) is loaded at a time.

> Legal: only cut content you own or have permission to re-edit.

## Setup (Windows & Linux)
1. Install **Python 3.10+**, **ffmpeg** (with libass; NVENC optional) and the NVIDIA driver + CUDA runtime.
2. `pip install -e .` then `pip install llama-cpp-python` (build with CUDA, see its docs) **or** install Ollama.
3. Put a bold font (e.g. `Montserrat-ExtraBold.ttf`) in `assets/fonts/` (font name in `config.yaml` must match).
4. `clipper download-models` (Whisper), then put a GGUF (Qwen 7B Instruct Q4_K_M) in `models/`
   and set `llm.model_path`, or `ollama pull qwen2.5:7b-instruct` and set `llm.backend: ollama`.
5. `clipper doctor` checks everything.

## Usage
```
clipper run --url "https://youtube.com/watch?v=..." --clips 8 --min 25 --max 75
clipper run --file video.mp4 --lang id
clipper run --file video.mp4 --dry-run            # candidates only, no render
clipper render --work work/<name> --clips 1,3,5 --style tiktok
clipper run --file video.mp4 --no-cache --no-teaser
```
Flags: `--dry-run --no-cache --fresh --no-teaser --glossary glossary.txt --config other.yaml`.
Results: `output/<name>/clip_XX.mp4`, `clip_XX.json`, `summary.md`, `summary.csv`.
Add names/jargon to `glossary.txt` (one per line): used as Whisper prompt + fuzzy/LLM spelling fix.

## What is in Phase 1 (+1.5)
Ingest (yt-dlp/file), Whisper word timestamps, sentence segmentation, audio signals + YouTube heatmap,
LLM selection by **sentence ID** with JSON-schema output, code-side scoring/NMS/dedup/ending fix,
transcript correction, hook teaser, word-highlight `.ass` captions, center-crop or blur 9:16 render
(NVENC with libx264 fallback), JSON stage cache, `--dry-run`, `doctor`.
Not yet (Phase 2+): face/speaker tracking (`reframe.mode: face` falls back to center), jump cuts,
scene detection, split-screen, Web UI.

## Notes / limits
- Mixed ID/EN videos: use `language: auto`; Whisper may mis-detect language on code-switching, so force `id` or `en` if captions look wrong.
- Changing clip count, durations or weights does NOT re-run Whisper/LLM (cached in `work/<name>/*.json`).
- Tests: `pytest tests` (deterministic parts, no GPU).
- Benchmarks on your machine are not measured yet; record them after your first 30-60 min video.
