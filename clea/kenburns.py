"""Turn a still image into a short pan/zoom ("Ken Burns") video clip.

Reuses the zoompan approach from render.py's punch-in effect, but as a
standalone clip generator: each generated image becomes an ordinary .mp4
that the existing beat-sync/scoring/caption pipeline consumes exactly like
a real video clip. This is what lets "generate from a topic" reuse the
whole auto-edit engine instead of being a separate rendering path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Alternating directions so a sequence of generated images doesn't all move
# the same way; kept subtle (max ~1.15x) so upscale artifacts stay minor.
_DIRECTIONS = [
    "z='min(1+0.0018*on,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",       # push in, centered
    "z='max(1.15-0.0018*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",     # pull out, centered
    "z='min(1+0.0018*on,1.15)':x='if(gte(zoom,1.0),(iw-iw/zoom)*0.15,0)':y='ih/2-(ih/zoom/2)'",  # push in, pan right
    "z='min(1+0.0018*on,1.15)':x='if(gte(zoom,1.0),(iw-iw/zoom)*0.85,0)':y='ih/2-(ih/zoom/2)'",  # push in, pan left
]


def render_ken_burns_clip(
    image_path: str,
    out_path: str,
    duration: float,
    width: int,
    height: int,
    fps: int = 30,
    direction_index: int = 0,
) -> None:
    direction = _DIRECTIONS[direction_index % len(_DIRECTIONS)]
    frames = max(1, round(duration * fps))
    # Upscale first so zoompan has headroom without visible pixelation at max zoom.
    filt = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan={direction}:d={frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", image_path,
        "-t", f"{duration:.3f}",
        "-vf", filt,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (ken burns) failed:\n{proc.stderr[-2000:]}")


def render_ken_burns_clips(
    image_paths: list[str], out_dir: str, cfg,
    seconds_per_image: float = 4.0,
) -> list[str]:
    w, h, fps = cfg.video["width"], cfg.video["height"], cfg.video["fps"]
    out_paths = []
    for i, img in enumerate(image_paths):
        out_path = str(Path(out_dir) / f"kb_{i:02d}.mp4")
        render_ken_burns_clip(img, out_path, seconds_per_image, w, h, fps,
                              direction_index=i)
        out_paths.append(out_path)
    return out_paths
