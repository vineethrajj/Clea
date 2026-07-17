"""Speech-to-text via faster-whisper with word-level timestamps.

GPU policy (8GB VRAM ceiling): the model is loaded once, all clips are
transcribed SEQUENTIALLY, and the model is explicitly freed before any other
GPU-capable stage (LLM, NVENC render) runs. Device selection is automatic:
CUDA -> "medium" @ float16, otherwise CPU -> "small" @ int8, per config.yaml.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

from .config import Config
from .hardware import detect_cuda_whisper


@dataclass
class Word:
    start: float
    end: float
    text: str


def resolve_whisper_settings(cfg: Config) -> tuple[str, str, str]:
    """Return (model_name, device, compute_type) honouring auto-fallback."""
    wcfg = cfg.whisper
    device = wcfg["device"]
    if device == "auto":
        device = "cuda" if detect_cuda_whisper() else "cpu"
    if device == "cuda":
        return wcfg["model"], "cuda", wcfg["compute_type_gpu"]
    return wcfg["cpu_fallback_model"], "cpu", wcfg["compute_type_cpu"]


def transcribe_clips(paths: list[str], cfg: Config,
                     model_override: str | None = None) -> dict[str, list[Word]]:
    """Transcribe each clip (words with timestamps), keyed by clip path.

    Clips without decodable speech/audio simply map to an empty list.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — heavy optional dep
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    model_name, device, compute_type = resolve_whisper_settings(cfg)
    if model_override:
        model_name = model_override
    print(f"      whisper: model={model_name} device={device} ({compute_type})")

    from .render import clip_has_audio  # ffprobe helper; no circular import

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    results: dict[str, list[Word]] = {}
    try:
        for path in paths:
            words: list[Word] = []
            if not clip_has_audio(path):
                results[path] = words
                continue
            try:
                segments, _info = model.transcribe(
                    path,
                    word_timestamps=True,
                    vad_filter=True,          # skip music/silence-only regions
                    beam_size=5,
                )
                for seg in segments:
                    for w in seg.words or []:
                        text = w.word.strip()
                        if text:
                            words.append(Word(float(w.start), float(w.end), text))
            except Exception as exc:  # e.g. clip has no audio stream
                print(f"      (no transcript for {path}: {exc})")
            results[path] = words
    finally:
        # Free VRAM before the next GPU stage — never hold the model across stages.
        del model
        gc.collect()
    return results
