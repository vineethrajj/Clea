"""Generate synthetic test media for the Phase 1 pipeline.

Creates test_media/clips/ with three clips of deliberately different energy
(so the motion scorer has something to rank) and test_media/beat_track.wav,
a 120 BPM kick/hat pattern that librosa locks onto cleanly.

Usage:  python scripts/make_test_media.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "test_media" / "clips"


def make_clip(name: str, lavfi: str, seconds: float) -> None:
    out = CLIPS / name
    sep = ":" if "=" in lavfi else "="
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"{lavfi}{sep}size=1280x720:rate=30",
        "-t", str(seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(out),
    ], check=True)
    print(f"  wrote {out}")


SPEECH_TEXT = (
    "Regular dental checkups help prevent cavities and gum disease. "
    "In most cases, professional cleaning removes plaque that brushing misses. "
    "Ask your dentist how often you should schedule a visit."
)


def make_speech_clip(name: str, text: str) -> None:
    """Talking-head stand-in: synthesized voice over a moving background,
    so the Phase 2 caption pipeline has real speech to transcribe.
    Skipped when espeak-ng is not installed."""
    import shutil
    if not shutil.which("espeak-ng"):
        print("  (espeak-ng not found — skipping speech clip)")
        return
    wav = CLIPS.parent / "_speech.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav), text],
                   check=True)
    out = CLIPS / name
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-i", str(wav),
        "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
    ], check=True)
    wav.unlink()
    print(f"  wrote {out} (with speech)")


def make_beat_track(path: Path, bpm: float = 120.0, seconds: float = 40.0,
                    sr: int = 44100) -> None:
    t_beat = 60.0 / bpm
    n = int(seconds * sr)
    audio = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(7)

    def add(start_s: float, sig: np.ndarray, gain: float) -> None:
        i = int(start_s * sr)
        j = min(i + len(sig), n)
        if i < n:
            audio[i:j] += gain * sig[: j - i]

    # Kick: 60 Hz decaying sine with a pitch drop, on every beat.
    dur = int(0.12 * sr)
    tt = np.arange(dur) / sr
    kick = np.sin(2 * np.pi * (60 * tt - 30 * tt**2)) * np.exp(-tt * 28)
    # Hat: short noise burst on off-beats. Snare-ish burst every 2nd beat.
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 180)
    snare = rng.standard_normal(int(0.09 * sr)) * np.exp(-np.arange(int(0.09 * sr)) / sr * 55)

    beat = 0
    t = 0.0
    while t < seconds:
        add(t, kick, 0.9)
        if beat % 2 == 1:
            add(t, snare, 0.45)
        add(t + t_beat / 2, hat, 0.25)
        beat += 1
        t = beat * t_beat

    # Simple bassline so the track isn't pure percussion.
    tt_all = np.arange(n) / sr
    bass_freq = 55 * (1 + 0.5 * ((tt_all // (4 * t_beat)) % 2))
    audio += 0.12 * np.sin(2 * np.pi * bass_freq * tt_all)

    audio /= max(np.abs(audio).max(), 1e-9)
    sf.write(path, (audio * 0.85).astype(np.float32), sr)
    print(f"  wrote {path} ({bpm:.0f} BPM, {seconds:.0f}s)")


def main() -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    print("Generating test clips...")
    # High motion: constantly changing test pattern.
    make_clip("high_motion.mp4", "testsrc2", 12)
    # Continuous organic motion: mandelbrot zoom.
    make_clip("zoom_motion.mp4", "mandelbrot", 12)
    # Low energy: nearly static gradient (should be picked last).
    make_clip("static_boring.mp4", "gradients=speed=0.01", 12)
    # Speech clip for the caption pipeline (Phase 2).
    make_speech_clip("talking.mp4", SPEECH_TEXT)
    print("Generating beat track...")
    make_beat_track(ROOT / "test_media" / "beat_track.wav")
    print("Done. Try:\n  python -m clea edit --clips test_media/clips "
          "--audio test_media/beat_track.wav -o output/test_reel.mp4")


if __name__ == "__main__":
    main()
