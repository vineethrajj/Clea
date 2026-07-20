"""End-to-end edit pipeline shared by the CLI and the web server.

GPU policy: everything here is strictly sequential — whisper transcription
completes and frees its model before scoring/rendering begins. Callers that
can run jobs concurrently (the web server) must additionally serialise
pipeline runs and LLM calls behind one lock so two GPU stages never overlap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import EditPlan, build_edl, build_slots
from .audio import analyze_beats
from .config import Config
from .hardware import HardwareReport, probe
from .render import choose_encoder_args, render
from .scoring import analyze_clip

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class EditOptions:
    duration: float | None = None
    style: str = "informational"     # key into config styles: presets
    captions: bool | None = None     # None -> style default
    notes: bool | None = None
    keep_voice: bool | None = None
    no_xfade: bool = False
    hook_text: str | None = None     # big opening title burned over the reel
    seed: int = 42
    whisper_model: str | None = None
    force_ordering: str | None = None    # bypass the style's ordering (e.g. "chronological")
    scene_captions: dict[str, str] | None = None  # clip path -> on-screen note card text
                                                    # (used when there's no real speech to
                                                    # transcribe, e.g. AI-generated scenes)


@dataclass
class ResolvedStyle:
    pacing: dict
    captions: bool
    notes: bool
    keep_voice: bool
    ordering: str
    transitions: str
    punch_in: bool


def resolve_style(cfg: Config, opts: EditOptions) -> ResolvedStyle:
    styles = dict(cfg.get("styles") or {})
    if opts.style not in styles:
        raise ValueError(f"unknown style '{opts.style}' — available: {', '.join(styles)}")
    style = styles[opts.style]
    pacing = {**dict(cfg.pacing), **dict(style.get("pacing") or {})}
    defaults = dict(style.get("defaults") or {})

    def pick(explicit: bool | None, key: str) -> bool:
        return bool(defaults.get(key, False)) if explicit is None else explicit

    return ResolvedStyle(
        pacing=pacing,
        captions=pick(opts.captions, "captions"),
        notes=pick(opts.notes, "notes"),
        keep_voice=pick(opts.keep_voice, "keep_voice"),
        ordering=str(style.get("ordering", "hook_first")),
        transitions=str(style.get("transitions", "standard")),
        punch_in=bool(style.get("punch_in", False)),
    )


@dataclass
class EditResult:
    out_path: str
    encoder: str
    total_duration: float
    n_cuts: int
    n_xfades: int
    n_caption_words: int = 0
    n_notes: int = 0
    render_seconds: float = 0.0
    plan: EditPlan | None = None
    note_texts: list[str] = field(default_factory=list)


def collect_clips(clips_dir: str | Path) -> list[str]:
    p = Path(clips_dir)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in VIDEO_EXTS else []
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)


def _notes_from_scene_captions(plan: EditPlan, mapping: dict[str, str]) -> list:
    """One Note per contiguous run of a clip on the timeline, showing the
    caption supplied for that clip (used for AI-generated scenes, which have
    no real speech to transcribe). Relies on chronological-style assembly
    where a clip's segments are contiguous, not scattered by hook-first
    reordering — callers should set force_ordering='chronological'."""
    from .notes import Note

    notes: list[Note] = []
    tl_pos = 0.0
    current_clip: str | None = None
    seg_start = 0.0
    for e in plan.entries:
        if e.clip != current_clip:
            if current_clip is not None and current_clip in mapping:
                notes.append(Note(mapping[current_clip], seg_start, tl_pos))
            current_clip, seg_start = e.clip, tl_pos
        tl_pos += e.timeline_duration
    if current_clip is not None and current_clip in mapping:
        notes.append(Note(mapping[current_clip], seg_start, tl_pos))
    return notes


def run_edit(
    clip_paths: list[str],
    audio_path: str,
    out_path: str,
    cfg: Config,
    opts: EditOptions,
    progress=lambda stage, msg: None,
    hw: HardwareReport | None = None,
) -> EditResult:
    if not clip_paths:
        raise ValueError("no video clips provided")

    style = resolve_style(cfg, opts)
    if opts.force_ordering:
        style.ordering = opts.force_ordering

    if hw is None:
        progress("probe", "checking hardware")
        hw = probe(cfg.llm["ollama_url"])
    if not hw.ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    encoder, _ = choose_encoder_args(cfg, hw)

    progress("beats", f"analysing beats in {Path(audio_path).name}")
    grid = analyze_beats(audio_path)
    progress("beats", f"{grid.tempo:.1f} BPM, {len(grid.beat_times)} beats")

    words_by_clip = None
    if style.captions or style.notes:
        # Sequential GPU stage 1: whisper loads, runs, and frees its model
        # inside transcribe_clips before anything else touches the GPU.
        progress("transcribe", f"transcribing {len(clip_paths)} clip(s)")
        from .transcribe import transcribe_clips
        words_by_clip = transcribe_clips(clip_paths, cfg, opts.whisper_model)
        n_words = sum(len(v) for v in words_by_clip.values())
        progress("transcribe", f"{n_words} words recognised")

    progress("score", f"scoring {len(clip_paths)} clip(s) for motion")
    analyses = [
        analyze_clip(p, cfg.scoring["sample_fps"], cfg.scoring["analysis_width"])
        for p in clip_paths
    ]

    target = opts.duration if opts.duration else cfg.video["target_duration"]
    target = max(cfg.video["min_duration"], min(cfg.video["max_duration"], target))
    target = min(target, grid.duration)

    progress("assemble",
             f"assembling ~{target:.0f}s '{opts.style}' sequence ({style.ordering})")
    slots = build_slots(grid, target, style.pacing, transitions=style.transitions)
    if opts.no_xfade:
        for s in slots:
            if s.transition_out == "xfade":
                s.transition_out, s.fade_dur = "cut", 0.0
    plan = build_edl(analyses, slots, dict(cfg.scoring), seed=opts.seed,
                     ordering=style.ordering)

    ass_path = None
    n_caption_words = 0
    note_texts: list[str] = []
    if words_by_clip or opts.hook_text or opts.scene_captions:
        from .captions import remap_words, write_ass

        timeline_words = []
        if style.captions and words_by_clip:
            timeline_words = remap_words(plan, words_by_clip)
        n_caption_words = len(timeline_words)

        notes = []
        if style.notes and words_by_clip:
            # Sequential GPU stage 2 (optional LLM call inside).
            from .notes import build_notes
            notes = build_notes(words_by_clip, plan, cfg,
                                max_notes=cfg.notes["max_notes"],
                                use_llm=cfg.notes["use_llm"])
            note_texts = [n.text for n in notes]
        elif opts.scene_captions:
            notes = _notes_from_scene_captions(plan, opts.scene_captions)
            note_texts = [n.text for n in notes]

        if timeline_words or notes or opts.hook_text:
            ass_path = str(Path(out_path).with_suffix(".ass"))
            write_ass(timeline_words, cfg, ass_path, notes=notes,
                      hook_text=opts.hook_text)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    progress("render", f"rendering {cfg.video['width']}x{cfg.video['height']} via {encoder}"
             + (" (punch-in)" if style.punch_in else ""))
    t0 = time.time()
    used = render(plan, audio_path, out_path, cfg, hw,
                  ass_path=ass_path, keep_voice=style.keep_voice,
                  punch_in=style.punch_in)

    return EditResult(
        out_path=out_path,
        encoder=used,
        total_duration=plan.total_duration,
        n_cuts=len(plan.entries),
        n_xfades=sum(1 for e in plan.entries if e.transition_out == "xfade"),
        n_caption_words=n_caption_words,
        n_notes=len(note_texts),
        render_seconds=time.time() - t0,
        plan=plan,
        note_texts=note_texts,
    )
