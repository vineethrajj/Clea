"""ffmpeg export: single-pass filter_complex render of the edit plan.

Output is 1080x1920 (9:16) — sources are scaled to cover and centre-cropped.
Encoding uses h264_nvenc when the hardware probe says it works, otherwise
libx264. All segments are normalised to the same fps/size/pixfmt/timebase so
concat and xfade can be chained freely.

Optional extras:
  * ass_path   — burn styled captions into the video (libass)
  * keep_voice — carry the source clips' own audio (hard-cut per segment)
                 mixed over the music, with the music ducked
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

from .assemble import EditPlan
from .config import Config
from .hardware import HardwareReport


def choose_encoder_args(cfg: Config, hw: HardwareReport) -> tuple[str, list[str]]:
    mode = cfg.encoder["mode"]
    if mode == "nvenc" or (mode == "auto" and hw.nvenc):
        return "h264_nvenc", list(cfg.encoder["nvenc_args"])
    return "libx264", list(cfg.encoder["cpu_args"])


def _escape_filter_path(path: str) -> str:
    """Escape a filename for use inside a filter argument (ass=...)."""
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


@lru_cache(maxsize=None)
def clip_has_audio(path: str) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def build_ffmpeg_command(
    plan: EditPlan,
    audio_path: str,
    out_path: str,
    cfg: Config,
    hw: HardwareReport,
    ass_path: str | None = None,
    keep_voice: bool = False,
) -> list[str]:
    w, h, fps = cfg.video["width"], cfg.video["height"], cfg.video["fps"]
    total = plan.total_duration

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for e in plan.entries:
        cmd += ["-i", e.clip]
    cmd += ["-i", audio_path]
    audio_idx = len(plan.entries)

    filters: list[str] = []
    for i, e in enumerate(plan.entries):
        filters.append(
            f"[{i}:v]trim=start={e.src_start:.4f}:duration={e.src_duration:.4f},"
            f"setpts=PTS-STARTPTS,fps={fps},"
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,format=yuv420p,settb=AVTB[s{i}]"
        )

    # Chain segments left-to-right. Crossfades use the fade extension sourced
    # in assemble.py, so `offset` is simply the timeline boundary and every
    # later cut stays on the beat.
    video_out = "[vout]" if not ass_path else "[vpre]"
    current = "[s0]"
    boundary = plan.entries[0].timeline_duration
    for i, e in enumerate(plan.entries[1:], start=1):
        prev = plan.entries[i - 1]
        out_label = video_out if i == len(plan.entries) - 1 else f"[c{i}]"
        if prev.transition_out == "xfade" and prev.fade_dur > 0:
            filters.append(
                f"{current}[s{i}]xfade=transition=fade:"
                f"duration={prev.fade_dur:.4f}:offset={boundary:.4f}{out_label}"
            )
        else:
            filters.append(f"{current}[s{i}]concat=n=2:v=1:a=0{out_label}")
        current = out_label
        boundary += e.timeline_duration
    if len(plan.entries) == 1:
        filters.append(f"{current}null{video_out}")
        current = video_out
    if ass_path:
        filters.append(f"[vpre]ass='{_escape_filter_path(ass_path)}'[vout]")

    acfg = cfg.audio_mix
    music_vol = acfg["music_volume_with_voice"] if keep_voice else acfg["music_volume"]
    fade_out = min(0.8, total / 4)
    music_chain = (
        f"[{audio_idx}:a]atrim=0:{total:.4f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:d=0.15,afade=t=out:st={max(total - fade_out, 0):.4f}:d={fade_out:.4f},"
        f"volume={music_vol}"
    )
    if keep_voice:
        # Voice always hard-cuts exactly on the timeline grid (no fade
        # extension), independent of the video transition type.
        voice_labels = []
        for i, e in enumerate(plan.entries):
            if clip_has_audio(e.clip):
                filters.append(
                    f"[{i}:a]atrim=start={e.src_start:.4f}:"
                    f"duration={e.timeline_duration:.4f},asetpts=PTS-STARTPTS,"
                    f"aformat=sample_rates=48000:channel_layouts=stereo,"
                    f"apad=whole_dur={e.timeline_duration:.4f}[va{i}]"
                )
            else:
                filters.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=0:{e.timeline_duration:.4f}[va{i}]"
                )
            voice_labels.append(f"[va{i}]")
        filters.append(
            "".join(voice_labels)
            + f"concat=n={len(voice_labels)}:v=0:a=1,"
            + f"volume={acfg['voice_volume']}[voice]"
        )
        filters.append(music_chain + "[music]")
        filters.append("[voice][music]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        filters.append(music_chain + "[aout]")

    encoder, enc_args = choose_encoder_args(cfg, hw)
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        *enc_args,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        *cfg.encoder["audio_args"],
        "-movflags", "+faststart",
        "-t", f"{total:.4f}",
        out_path,
    ]
    return cmd


def render(
    plan: EditPlan,
    audio_path: str,
    out_path: str,
    cfg: Config,
    hw: HardwareReport,
    ass_path: str | None = None,
    keep_voice: bool = False,
) -> str:
    if not plan.entries:
        raise ValueError("Edit plan is empty — nothing to render.")
    cmd = build_ffmpeg_command(plan, audio_path, out_path, cfg, hw,
                               ass_path=ass_path, keep_voice=keep_voice)
    encoder, _ = choose_encoder_args(cfg, hw)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({encoder}):\n{proc.stderr[-4000:]}")
    return encoder
