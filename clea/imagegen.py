"""Local text-to-image generation for the "no raw footage" path.

Deliberately illustrative-only, never photorealistic-people: local diffusion
on an 8GB laptop GPU cannot produce convincing photoreal humans, and for a
real dental clinic, AI-fabricated "patients" or "procedures" is a compliance
and trust problem regardless of quality. Every prompt is routed through a
fixed illustrative style and checked against a realism/person blocklist
before generation; the model isn't asked to depict people at all.

GPU policy: same sequential rule as whisper/LLM — one caller at a time,
pipeline must never run this concurrently with another GPU stage. The
pipeline object is created per batch and released afterward so VRAM is
freed before the next stage (render/NVENC) starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config

STYLE_SUFFIX = (
    "flat vector illustration, clean icon style, minimalist, soft "
    "gradient background, no photorealism, no human faces, no text"
)

NEGATIVE_PROMPT = (
    "photo, photorealistic, realistic, photograph, human face, person, "
    "patient, doctor, portrait, selfie, hyperrealistic, deformed, disfigured, "
    "text, watermark, logo"
)

# Prompts requesting realistic depictions of people/patients/procedures are
# rejected outright rather than silently reinterpreted — silent rewriting
# could give a false impression the request was honoured as asked.
_BLOCKED_PATTERNS = [
    r"\bpatient(s)?\b", r"\breal(istic)?\s+(person|people|human|face)\b",
    r"\bphoto(graph)?\s+of\s+a?\s*(person|patient|doctor|dentist)\b",
    r"\bselfie\b", r"\bportrait\b", r"\bmy\s+(clinic|office|team)\b",
    r"\bbefore\s*/?\s*after\b", r"\bsurgery\b", r"\bblood\b",
]


class PromptRejected(ValueError):
    pass


@dataclass
class GeneratedImage:
    prompt: str
    path: str


def build_safe_prompt(topic_prompt: str) -> str:
    """Raise if the prompt asks for realistic people/patients/procedures;
    otherwise wrap it in the fixed illustrative style."""
    lowered = topic_prompt.lower()
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            raise PromptRejected(
                f"Prompt looks like it's asking for realistic people, patients, "
                f"or procedure imagery ('{topic_prompt}'). Clea only generates "
                f"illustrative/abstract visuals — describe a concept instead, "
                f"e.g. 'a tooth with a shield icon representing protection'.")
    return f"{topic_prompt.strip()}, {STYLE_SUFFIX}"


_pipeline = None
_pipeline_model = None


def _load_pipeline(cfg: Config):
    global _pipeline, _pipeline_model
    import torch
    from diffusers import AutoPipelineForText2Image

    model = cfg.imagegen["model"]
    if _pipeline is not None and _pipeline_model == model:
        return _pipeline

    device = cfg.imagegen["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)
    _pipeline, _pipeline_model = pipe, model
    return pipe


def unload_pipeline() -> None:
    """Free VRAM before the next GPU stage (render/whisper/LLM) runs."""
    global _pipeline, _pipeline_model
    if _pipeline is not None:
        try:
            import torch
            del _pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    _pipeline, _pipeline_model = None, None


def generate_images(prompts: list[str], out_dir: str, cfg: Config,
                    progress=lambda i, n, msg: None) -> list[GeneratedImage]:
    """Generate one illustrative image per prompt, sequentially.

    Raises PromptRejected if any prompt matches the realism/person blocklist
    (checked for all prompts up front, before any GPU work starts).
    """
    from pathlib import Path

    safe_prompts = [build_safe_prompt(p) for p in prompts]  # validate all first

    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AI image generation needs 'torch' and 'diffusers'. Install with:\n"
            "  pip install diffusers accelerate\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
            "  (or the CPU wheel: --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pipe = _load_pipeline(cfg)
    icfg = cfg.imagegen
    results: list[GeneratedImage] = []
    try:
        for i, (raw, safe) in enumerate(zip(prompts, safe_prompts)):
            progress(i + 1, len(prompts), f"generating image: {raw[:50]}")
            kwargs = dict(
                prompt=safe,
                num_inference_steps=icfg["steps"],
                width=icfg["width"],
                height=icfg["height"],
            )
            if icfg["guidance_scale"] > 0:
                kwargs["guidance_scale"] = icfg["guidance_scale"]
                kwargs["negative_prompt"] = NEGATIVE_PROMPT
            else:
                kwargs["guidance_scale"] = 0.0  # turbo-style models: no CFG
            image = pipe(**kwargs).images[0]
            path = str(Path(out_dir) / f"img_{i:02d}.png")
            image.save(path)
            results.append(GeneratedImage(prompt=raw, path=path))
    finally:
        unload_pipeline()
    return results
