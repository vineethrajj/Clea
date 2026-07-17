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
| 3 | Informational note-card overlays from transcript | ✅ working |
| 4 | AI content generator (Ollama qwen2.5:7b, dental compliance guardrail) | ✅ working |
| 5 | Mobile-friendly web UI (FastAPI + vanilla JS, phone over LAN) | ✅ working |
| 5.5 | Dental reel style presets + engagement features + regression suite | ✅ working |
| 6 | Launcher scripts (double-click setup/run); full installer packaging later | ✅ scripts |

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

## Phase 3 usage — note-card overlays

```bash
python -m clea edit --clips myclips --audio song.mp3 -o output/reel.mp4 \
    --captions --notes --keep-voice
```

`--notes` picks 2–4 key factual sentences from the transcript (numbers,
informational verbs, sensible length), compresses them via local Ollama when
it's running (plain trimming otherwise), and overlays them as boxed cards at
the top of the frame, timed to when each sentence is spoken. Style under
`notes:` in `config.yaml`.

## Phase 4 usage — AI content generator

```bash
ollama pull qwen2.5:7b   # once

python -m clea generate --topic "ultrasonic scaling" --type dental
python -m clea generate --topic "database normalization" --type educational-other
```

Output: hook line, 4-part script (hook / problem / solution / CTA), caption,
5–8 hashtags. The **dental** type enforces the compliance guardrail in the
system prompt (no "instant/painless/best/cure/guaranteed", soft "in most
cases" framing, "consult your dentist") *and* post-checks the output against
a banned-word list, regenerating once on violation. `--json` for raw JSON.

## Phase 5 usage — web UI

```bash
python -m clea serve            # http://localhost:8000
```

Startup prints your LAN address (e.g. `http://192.168.1.23:8000`) — open it
on your phone over the same wifi. One page, two tabs:

- **Auto-Edit** — drag-drop (or tap-pick) clips + music, toggle captions /
  note cards / keep-voice / crossfades, 15–30s length slider, live progress,
  inline 9:16 preview, download button.
- **Content Ideas** — topic + type in, hook/script/caption/hashtags out,
  copy-to-clipboard per block.

Status chips in the header show at a glance whether NVENC, Whisper-GPU and
Ollama are active or something fell back to CPU. Jobs and LLM calls share
one lock, so GPU stages always run sequentially (8GB VRAM rule).

## Reel styles — tuned for dental Instagram

```bash
python -m clea edit --clips myclips --audio song.mp3 -o output/reel.mp4 \
    --style cool-edit --hook-text "SMILE UPGRADE"
```

| Style | Pacing | Transitions | Extras | Default overlays |
|---|---|---|---|---|
| `informational` | medium (0.6–3.0s cuts) | cut on strong beats, crossfade on weak | — | captions + notes + voice |
| `cool-edit` | fast (0.35–1.6s cuts) | **white-flash pop on strong beats** | **punch-in/out zoom every cut** | none (music-led) |
| `educational` | calm (0.7–3.5s cuts) | standard | — | captions + notes + voice |
| `case-study` | calm (0.8–3.5s cuts) | standard | **clip order preserved (before → after)** | captions + notes + voice |

- Style presets set pacing, transitions, ordering, and overlay defaults; any
  explicit `--captions/--no-captions`, `--notes/--no-notes`,
  `--keep-voice/--no-keep-voice` flag overrides the preset.
- `--hook-text "..."` burns a big yellow title over the first ~2s — pair it
  with a hook line from the content generator.
- All presets live under `styles:` in `config.yaml` — tweak or add your own.

The content generator mirrors these with `--format`
(`informational | cool-edit | educational | case-study | myth-bust`):

```bash
python -m clea generate --topic "smile makeover" --type dental --format case-study
```

Case-study packs are written anonymised ("in this case", "results vary");
cool-edit packs come as short on-screen text lines (`overlay_texts`) instead
of narration. The dental compliance guardrail applies to every format.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

45 tests: beat-grid/slot invariants (cuts land on beats, pacing bounds),
EDL properties (hook-first ordering, chronological mode, no source reuse,
crossfade timeline preservation), ASS caption/note/hook generation, ffmpeg
command construction (NVENC vs CPU args, xfade offsets, keep-voice mix,
punch-in zoompan), compliance checker, style resolution, and end-to-end
renders of every style preset verified with ffprobe. Run them after any
change to the edit engine.

## Quick start scripts

- Windows: `scripts\setup_windows.bat` once, then `scripts\run_windows.bat`
  (opens the browser automatically).
- Linux/macOS: `./scripts/run_linux.sh`.

Full single-file installer packaging (PyInstaller/Tauri) and any cloud
deployment are intentionally deferred — cloud hosting would add real
compute costs vs. the free local-only setup.

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
