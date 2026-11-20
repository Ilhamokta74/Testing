from __future__ import annotations

from pathlib import Path

from .gpu import GPU, is_oom
from .util import ClipperError


def _load(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ClipperError("faster-whisper is not installed: pip install faster-whisper") from e
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _run(model, wav: Path, language: str | None, glossary: list[str], progress):
    kwargs = dict(word_timestamps=True, vad_filter=True, language=language,
                  condition_on_previous_text=False)
    if glossary:
        prompt = ", ".join(glossary)[:400]
        kwargs["initial_prompt"] = prompt
        try:
            segments, info = model.transcribe(str(wav), hotwords=" ".join(glossary)[:400], **kwargs)
        except TypeError:  # older faster-whisper without hotwords
            segments, info = model.transcribe(str(wav), **kwargs)
    else:
        segments, info = model.transcribe(str(wav), **kwargs)

    words = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append({"w": text, "start": round(w.start, 3),
                              "end": round(w.end, 3), "p": round(w.probability, 3)})
        if progress and info.duration:
            progress(min(0.99, seg.end / info.duration), "Transcribing")
    return {"language": info.language, "duration": info.duration, "words": words}


def transcribe(wav: Path, model_name: str, compute_type: str, language: str,
               glossary: list[str], progress=None, device: str = "auto") -> dict:
    lang = None if language == "auto" else language
    if device == "cpu":
        attempts = [("cpu", "int8")]
    else:  # auto | cuda: try NVIDIA first, then CPU
        attempts = [("cuda", compute_type), ("cuda", "int8"), ("cpu", "int8")]
    last_err: Exception | None = None
    for device, ctype in attempts:
        try:
            with GPU.slot("whisper") if device == "cuda" else _nullctx():
                model = _load(model_name, device, ctype)
                try:
                    return _run(model, wav, lang, glossary, progress)
                finally:
                    del model
        except Exception as e:  # noqa: BLE001
            last_err = e
            if is_oom(e) or device == "cuda":
                if progress:
                    progress(0.0, f"GPU failed ({device}/{ctype}): {str(e)[:80]}. Falling back.")
                continue
            raise
    raise ClipperError(f"Transcription failed on all backends: {last_err}")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
