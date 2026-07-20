"""Phase 7: generate a full reel from a topic, no raw footage required.

topic -> LLM content pack (hook/script/caption/hashtags) -> a handful of
illustrative image prompts -> locally generated images -> Ken Burns pan/zoom
clips -> the same beat-sync/style/caption pipeline used for real footage.

Scene order is preserved (forced chronological) so each on-screen note card
matches the image actually showing at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .content import ContentPack, generate_content
from .hardware import HardwareReport
from .imagegen import GeneratedImage, generate_images
from .kenburns import render_ken_burns_clips
from .pipeline import EditOptions, EditResult, run_edit


@dataclass
class ImagineOptions:
    topic: str
    content_type: str = "educational-other"
    reel_format: str = "informational"
    style: str = "informational"       # edit style preset (pacing/transitions)
    music_id: str | None = None
    audio_path: str | None = None
    duration: float | None = None
    show_scene_captions: bool = True   # burn each scene's line as a note card
    seed: int = 42


@dataclass
class ImagineResult:
    pack: ContentPack
    image_prompts: list[str] = field(default_factory=list)
    edit: EditResult | None = None


def scene_prompts(pack: ContentPack, max_scenes: int) -> list[str]:
    prompts: list[str] = list(pack.overlay_texts) if pack.overlay_texts else (
        [pack.hook] + [v for v in pack.script.values() if v]
    )
    seen: set[str] = set()
    unique: list[str] = []
    for p in prompts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p.strip())
    return unique[:max_scenes] or [pack.topic]


def generate_video_from_topic(
    opts: ImagineOptions,
    cfg: Config,
    workdir: str,
    out_path: str,
    progress=lambda stage, msg: None,
    hw: HardwareReport | None = None,
) -> ImagineResult:
    if not opts.audio_path and not opts.music_id:
        raise ValueError("provide either audio_path or music_id")

    progress("script", f"writing content pack for '{opts.topic}'")
    pack = generate_content(opts.topic, opts.content_type, cfg,
                            reel_format=opts.reel_format)

    max_scenes = int(cfg.imagegen.get("max_scenes", 5))
    prompts = scene_prompts(pack, max_scenes)

    progress("images", f"generating {len(prompts)} illustrative image(s) locally")
    images_dir = str(Path(workdir) / "images")
    images: list[GeneratedImage] = generate_images(
        prompts, images_dir, cfg,
        progress=lambda i, n, msg: progress("images", f"{i}/{n}: {msg}"))

    progress("kenburns", "rendering pan/zoom clips from the generated images")
    clips_dir = str(Path(workdir) / "clips")
    clip_paths = render_ken_burns_clips(
        [im.path for im in images], clips_dir, cfg,
        seconds_per_image=float(cfg.imagegen["seconds_per_image"]))

    audio_path = opts.audio_path
    if opts.music_id:
        from .music_library import track_path
        p = track_path(cfg, opts.music_id)
        if not p:
            raise ValueError(f"unknown music_id: {opts.music_id}")
        audio_path = str(p)

    scene_captions = None
    if opts.show_scene_captions:
        scene_captions = {clip: im.prompt for clip, im in zip(clip_paths, images)}

    edit_opts = EditOptions(
        duration=opts.duration, style=opts.style,
        captions=False, notes=False, keep_voice=False,  # no real speech to transcribe
        hook_text=pack.hook, seed=opts.seed,
        force_ordering="chronological",   # scene order must match on-screen text
        scene_captions=scene_captions,
    )
    edit_result = run_edit(clip_paths, audio_path, out_path, cfg, edit_opts,
                           progress=progress, hw=hw)
    return ImagineResult(pack=pack, image_prompts=[im.prompt for im in images],
                         edit=edit_result)
