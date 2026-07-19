"""Local music library: scan a folder of tracks, auto-tag BPM/duration, and
expose a pick-from-list experience (like a Reels music picker) instead of
requiring a fresh upload every time.

No real commercial/trending audio is bundled or fetched — Instagram's music
catalog is licensed by Meta and can't be legally redistributed by a local
tool. Instead this scans whatever royalty-free/CC/your-own tracks you drop
into the library folder. See README for where to legally source free music.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import Config

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

# (title, bpm, mood tags, seed) — simple procedural synth: kick+hat pattern
# plus a sine/saw pad, varied by tempo and a couple of timbral knobs. Not
# copyrighted audio; a placeholder library, not a substitute for real music.
_STARTER_TRACKS = [
    ("Upbeat Pop Starter", 128.0, ["upbeat", "pop"], 1),
    ("Chill Lofi Starter", 90.0, ["chill", "lofi"], 2),
    ("Cinematic Build Starter", 100.0, ["cinematic", "build"], 3),
    ("High Energy Starter", 150.0, ["energetic", "workout"], 4),
]


def _synth_track(bpm: float, seconds: float, seed: int, sr: int = 44100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_beat = 60.0 / bpm
    n = int(seconds * sr)
    audio = np.zeros(n, dtype=np.float64)

    dur = int(0.12 * sr)
    tt = np.arange(dur) / sr
    kick = np.sin(2 * np.pi * (58 * tt - 26 * tt**2)) * np.exp(-tt * 26)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 190)
    snare = rng.standard_normal(int(0.09 * sr)) * np.exp(-np.arange(int(0.09 * sr)) / sr * 55)

    beat = 0
    t = 0.0
    while t < seconds:
        i = int(t * sr)
        j = min(i + len(kick), n)
        if i < n:
            audio[i:j] += 0.85 * kick[: j - i]
        hi = int((t + t_beat / 2) * sr)
        hj = min(hi + len(hat), n)
        if hi < n:
            audio[hi:hj] += 0.22 * hat[: hj - hi]
        if beat % 4 == 2:
            si = int(t * sr)
            sj = min(si + len(snare), n)
            if si < n:
                audio[si:sj] += 0.4 * snare[: sj - si]
        beat += 1
        t = beat * t_beat

    tt_all = np.arange(n) / sr
    root = 55.0 * (1 + 0.25 * ((tt_all // (4 * t_beat)) % 3))
    pad = 0.10 * np.sin(2 * np.pi * root * tt_all) + 0.05 * np.sin(2 * np.pi * root * 1.5 * tt_all)
    audio += pad

    peak = np.abs(audio).max()
    return (audio / peak * 0.82).astype(np.float32) if peak > 0 else audio.astype(np.float32)


def generate_starter_pack(tracks_dir: Path, seconds: float = 30.0) -> None:
    tracks_dir.mkdir(parents=True, exist_ok=True)
    for title, bpm, _tags, seed in _STARTER_TRACKS:
        filename = title.lower().replace(" ", "_") + ".wav"
        path = tracks_dir / filename
        if path.exists():
            continue
        audio = _synth_track(bpm, seconds, seed)
        sf.write(path, audio, 44100)


@dataclass
class Track:
    id: str
    filename: str
    title: str
    tempo: float
    duration: float
    tags: list[str] = field(default_factory=list)
    license: str = ""
    added_at: float = 0.0


def _library_paths(cfg: Config) -> tuple[Path, Path]:
    lib = cfg.music_library
    tracks_dir = Path(lib["dir"])
    manifest_path = Path(lib["manifest"])
    tracks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return tracks_dir, manifest_path


def load_manifest(cfg: Config) -> list[Track]:
    _, manifest_path = _library_paths(cfg)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Track(**t) for t in data]


def save_manifest(cfg: Config, tracks: list[Track]) -> None:
    _, manifest_path = _library_paths(cfg)
    manifest_path.write_text(
        json.dumps([asdict(t) for t in tracks], indent=2), encoding="utf-8")


def _title_from_filename(name: str) -> str:
    return name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()


def scan_library(cfg: Config) -> list[Track]:
    """Detect new/changed audio files and (re)tag them with tempo+duration.

    Existing entries (matched by filename) keep their manually-edited title/
    tags/license unless the underlying file was replaced.
    """
    tracks_dir, _ = _library_paths(cfg)
    existing = {t.filename: t for t in load_manifest(cfg)}
    on_disk = {p.name for p in tracks_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS}

    result: list[Track] = []
    for filename in sorted(on_disk):
        path = tracks_dir / filename
        prior = existing.get(filename)
        mtime = path.stat().st_mtime
        if prior and prior.added_at >= mtime:
            result.append(prior)
            continue
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        duration = float(len(y) / sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        result.append(Track(
            id=prior.id if prior else uuid.uuid4().hex[:10],
            filename=filename,
            title=prior.title if prior else _title_from_filename(filename),
            tempo=round(float(np.atleast_1d(tempo)[0]), 1),
            duration=round(duration, 1),
            tags=prior.tags if prior else [],
            license=prior.license if prior else "",
            added_at=mtime,
        ))
    # Drop entries whose file no longer exists.
    save_manifest(cfg, result)
    return result


def track_path(cfg: Config, track_id: str) -> Path | None:
    tracks_dir, _ = _library_paths(cfg)
    for t in load_manifest(cfg):
        if t.id == track_id:
            p = tracks_dir / t.filename
            return p if p.exists() else None
    return None


def ensure_starter_pack(cfg: Config) -> list[Track]:
    """Generate a few procedural, license-free tracks so the picker isn't
    empty on first run. These are code-synthesized (not copyrighted audio),
    tagged clearly as a starter pack — swap in real royalty-free tracks
    (Pixabay Music / YouTube Audio Library / Incompetech) for real posting.
    """
    tracks_dir, _ = _library_paths(cfg)
    if any(p.suffix.lower() in AUDIO_EXTS for p in tracks_dir.iterdir()):
        return load_manifest(cfg)
    generate_starter_pack(tracks_dir)
    tracks = scan_library(cfg)
    by_name = {t.filename: t for t in tracks}
    for title, _bpm, tags, _seed in _STARTER_TRACKS:
        filename = title.lower().replace(" ", "_") + ".wav"
        if filename in by_name:
            by_name[filename].title = title
            by_name[filename].tags = tags
            by_name[filename].license = "Clea starter pack (procedurally generated, no license needed)"
    save_manifest(cfg, tracks)
    return tracks
