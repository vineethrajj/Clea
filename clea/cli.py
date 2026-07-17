"""Phase 1 CLI: beat-sync auto-cut engine.

    python -m clea doctor                     # hardware / dependency check
    python -m clea beats  --audio track.mp3   # inspect detected tempo + beats
    python -m clea edit   --clips DIR --audio track.mp3 -o reel.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def cmd_doctor(args: argparse.Namespace) -> int:
    from .config import load_config
    from .hardware import format_report, probe

    cfg = load_config(args.config)
    report = probe(cfg.llm["ollama_url"])
    print(format_report(report))
    return 0 if report.ffmpeg else 1


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
    from .assemble import build_edl, build_slots
    from .audio import analyze_beats
    from .config import load_config
    from .hardware import probe
    from .render import choose_encoder_args, render
    from .scoring import analyze_clip

    cfg = load_config(args.config)
    clips_dir = Path(args.clips)
    clip_paths = sorted(
        str(p) for p in clips_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    ) if clips_dir.is_dir() else []
    if not clip_paths:
        print(f"error: no video clips found in {clips_dir}", file=sys.stderr)
        return 1

    print("[1/5] Hardware probe...")
    hw = probe(cfg.llm["ollama_url"])
    if not hw.ffmpeg:
        print("error: ffmpeg not found on PATH — install it first.", file=sys.stderr)
        return 1
    encoder, _ = choose_encoder_args(cfg, hw)
    print(f"      encoder: {encoder}"
          + ("" if encoder == "h264_nvenc" else "  (NVENC unavailable — CPU fallback)"))

    print(f"[2/5] Beat analysis: {args.audio}")
    grid = analyze_beats(args.audio)
    print(f"      {grid.tempo:.1f} BPM, {len(grid.beat_times)} beats over {grid.duration:.1f}s")

    words_by_clip = None
    if args.captions:
        # GPU stages run sequentially: whisper finishes and frees its model
        # (see transcribe_clips) before the NVENC render starts.
        print(f"[2b]  Transcribing {len(clip_paths)} clip(s) with faster-whisper...")
        from .transcribe import transcribe_clips
        words_by_clip = transcribe_clips(clip_paths, cfg, args.whisper_model)
        n_words = sum(len(v) for v in words_by_clip.values())
        print(f"      {n_words} words recognised")
        if not n_words:
            print("      (no speech found — export will have no captions)")

    print(f"[3/5] Scoring {len(clip_paths)} clip(s) for motion/interest...")
    analyses = []
    for p in clip_paths:
        a = analyze_clip(p, cfg.scoring["sample_fps"], cfg.scoring["analysis_width"])
        analyses.append(a)
        print(f"      {Path(p).name}: {a.duration:.1f}s, "
              f"mean score {a.scores.mean():.3f}, peak {a.scores.max():.3f}")

    target = args.duration if args.duration else cfg.video["target_duration"]
    target = max(cfg.video["min_duration"], min(cfg.video["max_duration"], target))
    target = min(target, grid.duration)

    print(f"[4/5] Assembling beat-cut sequence (~{target:.0f}s)...")
    slots = build_slots(grid, target, dict(cfg.pacing))
    if args.no_xfade:
        for s in slots:
            if s.transition_out == "xfade":
                s.transition_out, s.fade_dur = "cut", 0.0
    plan = build_edl(analyses, slots, dict(cfg.scoring), seed=args.seed)
    n_fades = sum(1 for e in plan.entries if e.transition_out == "xfade")
    print(f"      {len(plan.entries)} cuts ({n_fades} crossfades), "
          f"total {plan.total_duration:.2f}s")
    for i, e in enumerate(plan.entries):
        print(f"      #{i + 1:>2} {Path(e.clip).name} "
              f"src {e.src_start:6.2f}s +{e.timeline_duration:.2f}s "
              f"score {e.score:.3f} -> {e.transition_out}")

    out_path = args.output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ass_path = None
    if args.captions and words_by_clip:
        from .captions import remap_words, write_ass
        timeline_words = remap_words(plan, words_by_clip)
        if timeline_words:
            ass_path = str(Path(out_path).with_suffix(".ass"))
            write_ass(timeline_words, cfg, ass_path)
            print(f"      captions: {len(timeline_words)} words on timeline -> {ass_path}")
        else:
            print("      captions: no transcribed words landed in the chosen segments")

    print(f"[5/5] Rendering {cfg.video['width']}x{cfg.video['height']} via {encoder}"
          + (" + captions" if ass_path else "")
          + (" + source voice" if args.keep_voice else "") + "...")
    t0 = time.time()
    used = render(plan, args.audio, out_path, cfg, hw,
                  ass_path=ass_path, keep_voice=args.keep_voice)
    print(f"done: {out_path} ({used}, {time.time() - t0:.1f}s render)")
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
    p_edit.add_argument("--seed", type=int, default=42)
    p_edit.add_argument("--no-xfade", action="store_true", help="hard cuts only")
    p_edit.add_argument("--captions", action="store_true",
                        help="transcribe clip speech and burn in word-highlight captions")
    p_edit.add_argument("--keep-voice", action="store_true",
                        help="keep source clip audio audible, duck the music under it")
    p_edit.add_argument("--whisper-model", default=None,
                        help="override whisper model size (tiny/base/small/medium/large-v3)")

    p_tr = sub.add_parser("transcribe", help="debug: print word timestamps for clips")
    p_tr.add_argument("--clips", required=True, help="folder of clips or a single file")
    p_tr.add_argument("--whisper-model", default=None)

    args = parser.parse_args(argv)
    return {"doctor": cmd_doctor, "beats": cmd_beats, "edit": cmd_edit,
            "transcribe": cmd_transcribe}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
