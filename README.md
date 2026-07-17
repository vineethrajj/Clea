# Clea — local beat-sync video auto-editor + AI content generator

A Beat.ly-style auto-editor that runs **entirely free, entirely on your own
machine**: cuts your clips to the music's beat with high-engagement Reels
pacing, exports 9:16 via NVENC hardware encoding, and (in later phases) adds
Whisper captions, informational note-card overlays, and a local-LLM content
generator for dental-clinic marketing and exam-prep content.

Tuned for: **ASUS ROG Strix 16 — i9, 16GB RAM, RTX 4070 Laptop (8GB VRAM)**.
All GPU stages (LLM, Whisper) run **sequentially, never concurrently**, so
nothing fights over the 8GB VRAM ceiling. Every GPU feature has an automatic
CPU fallback, so the same code runs (slower) on any machine.

## Status

| Phase | Feature | Status |
|---|---|---|
| 1 | Beat-sync auto-cut CLI (librosa beats + OpenCV scoring + NVENC export) | ✅ working |
| 2 | Burned-in captions (faster-whisper "medium" on CUDA, ASS word-highlight) | ✅ working |
| 3 | Informational note-card overlays from transcript | ⬜ |
| 4 | AI content generator (Ollama qwen2.5:7b, dental compliance guardrail) | ⬜ |
| 5 | Mobile-friendly web UI (FastAPI + vanilla JS, phone over LAN) | ⬜ |

## Setup (on the ROG laptop)

1. **Python 3.10+** and deps:
   ```
   pip install -r requirements.txt
   ```
2. **ffmpeg with NVENC** — on Windows grab a full build from
   https://www.gyan.dev/ffmpeg/builds/ (the "full" release includes
   `h264_nvenc`) and put it on PATH. On Linux, distro ffmpeg + NVIDIA driver
   is enough. Verify: `ffmpeg -encoders | findstr nvenc`.
3. **NVIDIA driver** — a current Game Ready/Studio driver provides both NVENC
   and CUDA. Nothing else to install for Phase 1.
4. **Ollama** (needed from Phase 4): install from https://ollama.com then
   `ollama pull qwen2.5:7b`.
5. Check everything is detected:
   ```
   python -m clea doctor
   ```
   Expected on the ROG laptop: ffmpeg ✅, NVIDIA GPU ✅ (RTX 4070 Laptop,
   8188 MB), NVENC ✅. Whisper-CUDA and Ollama show ✅ once Phases 2/4 deps
   are installed. If NVENC shows `[--]`, exports still work via libx264 —
   just slower.

## Phase 1 usage

```bash
# generate synthetic test clips + a 120 BPM track (or use your own)
python scripts/make_test_media.py

# inspect the beats the editor will cut on
python -m clea beats --audio test_media/beat_track.wav

# auto-edit: folder of clips + music -> 9:16 reel
python -m clea edit --clips test_media/clips --audio test_media/beat_track.wav \
    -o output/reel.mp4 --duration 20
```

Options: `--duration 15..30`, `--no-xfade` (hard cuts only), `--seed N`
(re-roll segment choices), `--config path/to/config.yaml`.

## Phase 2 usage — captions

```bash
pip install faster-whisper   # once

# captions off by default; add per export:
python -m clea edit --clips myclips --audio song.mp3 -o output/reel.mp4 \
    --captions --keep-voice

# debug: see exactly what whisper hears in your clips
python -m clea transcribe --clips myclips
```

- `--captions` transcribes every clip (word timestamps, VAD-filtered),
  remaps the words of each *used* segment onto the edit timeline, writes an
  `.ass` file next to the output, and burns it in — white bold text, black
  outline, currently-spoken word highlighted yellow, lower-third placement
  clear of the Reels UI. Style lives under `captions:` in `config.yaml`.
- `--keep-voice` keeps the source clips' own audio audible in the final mix
  (hard-cut on the timeline grid) and ducks the music to 25% under it —
  for talking-head/voiceover content where the captions caption real speech.
- `--whisper-model tiny|base|small|medium|large-v3` overrides the config
  default for one run.
- On the RTX 4070 the model auto-selects **medium on CUDA (float16)**;
  machines without CUDA drop to **small on CPU (int8)** automatically. The
  whisper model is loaded once, used, and freed *before* the render starts —
  GPU stages never overlap (8GB VRAM rule).

### How the edit works

1. `librosa.beat.beat_track` finds tempo + beat timestamps; onset strength at
   each beat classifies it **strong** (hard cut) or **weak** (0.2s crossfade).
2. OpenCV scores each clip's motion (frame differencing) + sharpness/contrast
   on downscaled greyscale frames — static, boring footage scores low.
3. Cut slots are built on the beat grid: the first ~6s ("hook") gets the
   fastest cuts (≥0.45s) and, because slots greedily claim the best-scored
   segments in order, the most interesting material lands first.
4. Crossfades source *extra* frames from the outgoing segment so every cut
   boundary still lands exactly on its beat.
5. One ffmpeg pass renders 1080x1920@30 (scale-to-cover + centre crop) with
   `h264_nvenc` when available, `libx264` otherwise, music trimmed and faded.

## Configuration

Everything lives in `config.yaml` — encoder mode (`auto`/`nvenc`/`cpu`),
pacing (hook length, min/max cut, crossfade), Whisper model
(`medium` GPU / `small` CPU fallback), and the LLM model name
(`qwen2.5:7b` by default; swap to `qwen3:14b` or `llama3.2:3b` without code
changes). An optional cloud API key (`CLEA_CLOUD_API_KEY`) can be set for
higher-quality LLM output, but nothing requires it.

## Privacy / cost

No cloud calls for core functionality. Video never leaves your machine.
Every dependency is free and open-source.
