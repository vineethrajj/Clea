"""Phase 1 CLI: beat-sync auto-cut engine.

    python -m clea doctor                     # hardware / dependency check
    python -m clea beats  --audio track.mp3   # inspect detected tempo + beats
    python -m clea edit   --clips DIR --audio track.mp3 -o reel.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def cmd_doctor(args: argparse.Namespace) -> int:
    from .config import load_config
    from .hardware import format_report, probe

    cfg = load_config(args.config)
    report = probe(cfg.llm["ollama_url"])
    print(format_report(report))
    return 0 if report.ffmpeg else 1


def cmd_music(args: argparse.Namespace) -> int:
    from .config import load_config
    from .music_library import ensure_starter_pack, scan_library

    cfg = load_config(args.config)
    if args.music_cmd == "scan":
        tracks = scan_library(cfg)
    else:  # list
        tracks = ensure_starter_pack(cfg)
    if not tracks:
        print("no tracks found — drop mp3/wav files into "
              f"{cfg.music_library['dir']} then run 'clea music scan'")
        return 0
    for t in tracks:
        tags = ",".join(t.tags) if t.tags else "-"
        print(f"  {t.id}  {t.title:<28s} {t.tempo:6.1f} BPM  {t.duration:6.1f}s  [{tags}]")
    return 0


def cmd_beats(args: argparse.Namespace) -> int:
    from .audio import analyze_beats

    grid = analyze_beats(args.audio)
    print(f"tempo:    {grid.tempo:.1f} BPM")
    print(f"duration: {grid.duration:.2f}s")
    print(f"beats:    {len(grid.beat_times)}")
    strong = grid.beat_strengths >= (grid.beat_strengths.mean() if len(grid.beat_strengths) else 0)
    for t, s, is_strong in list(zip(grid.beat_times, grid.beat_strengths, strong))[:40]:
        print(f"  {t:7.3f}s  strength={s:.2f}  {'STRONG' if is_strong else 'weak'}")
    if len(grid.beat_times) > 40:
        print(f"  ... {len(grid.beat_times) - 40} more")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    from .config import load_config
    from .pipeline import EditOptions, collect_clips, run_edit

    cfg = load_config(args.config)
    clip_paths = collect_clips(args.clips)
    if not clip_paths:
        print(f"error: no video clips found in {args.clips}", file=sys.stderr)
        return 1

    opts = EditOptions(
        duration=args.duration, style=args.style, captions=args.captions,
        notes=args.notes, keep_voice=args.keep_voice, no_xfade=args.no_xfade,
        hook_text=args.hook_text, seed=args.seed,
        whisper_model=args.whisper_model,
    )
    try:
        result = run_edit(clip_paths, args.audio, args.output, cfg, opts,
                          progress=lambda stage, msg: print(f"[{stage:>10s}] {msg}"))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for i, e in enumerate(result.plan.entries):
        print(f"   #{i + 1:>2} {Path(e.clip).name} src {e.src_start:6.2f}s "
              f"+{e.timeline_duration:.2f}s score {e.score:.3f} -> {e.transition_out}")
    for text in result.note_texts:
        print(f"   note: {text}")
    print(f"done: {result.out_path} ({result.encoder}, "
          f"{result.total_duration:.2f}s video, {result.n_cuts} cuts, "
          f"{result.render_seconds:.1f}s render)")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from .config import load_config
    from .transcribe import transcribe_clips

    cfg = load_config(args.config)
    path = Path(args.clips)
    paths = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS) \
        if path.is_dir() else [str(path)]
    if not paths:
        print(f"error: no video found at {path}", file=sys.stderr)
        return 1
    results = transcribe_clips(paths, cfg, args.whisper_model)
    for clip, words in results.items():
        print(f"\n{clip}: {len(words)} words")
        for w in words:
            print(f"  {w.start:7.2f}-{w.end:7.2f}  {w.text}")
    return 0


def cmd_imagine(args: argparse.Namespace) -> int:
    from .config import load_config
    from .imagegen import PromptRejected
    from .imagine import ImagineOptions, generate_video_from_topic

    cfg = load_config(args.config)
    opts = ImagineOptions(
        topic=args.topic, content_type=args.content_type,
        reel_format=args.reel_format, style=args.style,
        music_id=args.music_id, audio_path=args.audio,
        duration=args.duration, show_scene_captions=not args.no_scene_captions,
    )
    try:
        result = generate_video_from_topic(
            opts, cfg, workdir=str(Path(args.output).parent / "_imagine_work"),
            out_path=args.output,
            progress=lambda stage, msg: print(f"[{stage:>10s}] {msg}"))
    except (RuntimeError, ValueError, PromptRejected) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"scenes: {result.image_prompts}")
    print(f"done: {result.edit.out_path} ({result.edit.encoder}, "
          f"{result.edit.total_duration:.2f}s, {result.edit.n_cuts} cuts)")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    import json

    from .config import load_config
    from .content import format_pack, generate_content
    from .llm import LLMError, ollama_available

    cfg = load_config(args.config)
    uses_cloud = cfg.llm["cloud_provider"] and cfg.llm["cloud_api_key"]
    if not uses_cloud and not ollama_available(cfg):
        print("error: Ollama is not running (and no cloud provider configured).\n"
              f"Start it and pull the model:  ollama pull {cfg.llm['model']}",
              file=sys.stderr)
        return 1
    try:
        pack = generate_content(args.topic, args.content_type, cfg,
                                reel_format=args.reel_format)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(pack.to_dict(), indent=2) if args.json else format_pack(pack))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clea", description="Beat-sync video auto-editor")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check ffmpeg / NVENC / CUDA / Ollama")

    p_beats = sub.add_parser("beats", help="show detected beats for an audio file")
    p_beats.add_argument("--audio", required=True)

    p_edit = sub.add_parser("edit", help="auto-cut clips to the beat and export 9:16")
    p_edit.add_argument("--clips", required=True, help="folder of video clips")
    p_edit.add_argument("--audio", required=True, help="music track")
    p_edit.add_argument("-o", "--output", default="output/reel.mp4")
    p_edit.add_argument("--duration", type=float, default=None,
                        help="target seconds (15-30, default from config)")
    p_edit.add_argument("--style", default="informational",
                        choices=["informational", "cool-edit", "educational", "case-study"],
                        help="reel style preset (pacing/transitions/overlay defaults)")
    p_edit.add_argument("--hook-text", default=None,
                        help="big opening title burned over the first ~2 seconds")
    p_edit.add_argument("--seed", type=int, default=42)
    p_edit.add_argument("--no-xfade", action="store_true", help="hard cuts only")
    p_edit.add_argument("--captions", action=argparse.BooleanOptionalAction, default=None,
                        help="word-highlight captions (default: per style)")
    p_edit.add_argument("--notes", action=argparse.BooleanOptionalAction, default=None,
                        help="factual note-card overlays (default: per style)")
    p_edit.add_argument("--keep-voice", action=argparse.BooleanOptionalAction, default=None,
                        help="keep clip audio, duck music (default: per style)")
    p_edit.add_argument("--whisper-model", default=None,
                        help="override whisper model size (tiny/base/small/medium/large-v3)")

    p_tr = sub.add_parser("transcribe", help="debug: print word timestamps for clips")
    p_tr.add_argument("--clips", required=True, help="folder of clips or a single file")
    p_tr.add_argument("--whisper-model", default=None)

    p_gen = sub.add_parser("generate", help="AI content pack for a topic (local Ollama)")
    p_gen.add_argument("--topic", required=True)
    p_gen.add_argument("--type", dest="content_type", default="educational-other",
                       choices=["dental", "educational-other"])
    p_gen.add_argument("--format", dest="reel_format", default="informational",
                       choices=["informational", "cool-edit", "educational",
                                "case-study", "myth-bust"],
                       help="what kind of reel the pack is written for")
    p_gen.add_argument("--json", action="store_true", help="print raw JSON")

    p_srv = sub.add_parser("serve", help="run the web UI (phone-friendly, LAN)")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)

    p_music = sub.add_parser("music", help="manage the local pick-from-list music library")
    p_music.add_argument("music_cmd", choices=["scan", "list"],
                         help="scan: re-tag new/changed files; list: show library "
                              "(generates a starter pack if empty)")

    p_imagine = sub.add_parser(
        "imagine", help="generate a full reel from a topic — no raw footage needed "
                         "(local illustrative AI images + Ken Burns, not photoreal people)")
    p_imagine.add_argument("--topic", required=True)
    p_imagine.add_argument("--type", dest="content_type", default="educational-other",
                           choices=["dental", "educational-other"])
    p_imagine.add_argument("--format", dest="reel_format", default="informational",
                           choices=["informational", "cool-edit", "educational",
                                    "case-study", "myth-bust"])
    p_imagine.add_argument("--style", default="informational",
                           choices=["informational", "cool-edit", "educational", "case-study"])
    p_imagine.add_argument("--audio", default=None, help="music file (or use --music-id)")
    p_imagine.add_argument("--music-id", default=None, help="track id from 'clea music list'")
    p_imagine.add_argument("--duration", type=float, default=None)
    p_imagine.add_argument("--no-scene-captions", action="store_true",
                           help="don't burn each scene's line as an on-screen note card")
    p_imagine.add_argument("-o", "--output", default="output/imagined.mp4")

    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server import serve
        serve(args.host, args.port, args.config)
        return 0
    return {"doctor": cmd_doctor, "beats": cmd_beats, "edit": cmd_edit,
            "transcribe": cmd_transcribe, "generate": cmd_generate,
            "music": cmd_music, "imagine": cmd_imagine}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
