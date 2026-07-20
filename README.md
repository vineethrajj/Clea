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
| 5.6 | Music library (pick-from-list, like Reels' song picker) | ✅ working |
| 7 | AI image generation + Ken Burns — full reels with no raw footage | ✅ working (optional install) |
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

## Music library — pick a track instead of uploading every time

Instagram's own music catalog is licensed by Meta and can't be legally
redistributed by a local tool, so Clea doesn't bundle real commercial audio.
Instead it gives you a **pick-from-list picker** (like Reels' music screen)
backed by whatever's in `music_library/tracks/`:

```bash
python -m clea music list     # generates a 4-track starter pack on first run,
                               # then lists id / title / detected BPM / duration
python -m clea music scan     # re-tag after adding/removing files
```

- **Starter pack**: on first run (CLI or web), four short procedurally
  generated instrumentals are created (`upbeat pop`, `chill lofi`,
  `cinematic build`, `high energy`) — code-synthesized, not copyrighted audio,
  so there's nothing to clear. They're placeholders for testing the picker
  and pacing, not for actually posting.
- **For real posting**, drop your own royalty-free/licensed tracks (mp3/wav)
  into `music_library/tracks/` and run `clea music scan` — it auto-detects
  BPM and duration via librosa, same analysis the auto-cut engine already
  uses. Good free sources: Pixabay Music, YouTube Audio Library, Incompetech
  (CC-BY, credit Kevin MacLeod) — check each track's license terms before
  publishing.
- The web UI's Music step has two tabs: **Pick a track** (library, with
  inline play preview, tempo/mood tags, one-tap select) and **Upload my
  own** (the original file picker). Both feed the same edit pipeline.

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

## No raw footage? Generate a reel from a topic (Phase 7)

For when you don't have clips to edit — writes a script locally, generates
a handful of **illustrative** AI images, pans/zooms them into clips (Ken
Burns effect), then runs the exact same beat-sync/style/caption pipeline as
real footage.

**Deliberately not photorealistic, and never depicts people.** An 8GB laptop
GPU can't produce convincing photoreal humans locally, and for a real dental
clinic, AI-fabricated "patients" or "procedures" is a compliance and trust
problem regardless of image quality. So this only generates abstract/icon/
concept-art visuals — every prompt is checked against a realism/person
blocklist (rejects things like "photo of a patient", "portrait", "before/
after of a real person") and wrapped in a fixed illustrative style
(`clea/imagegen.py: STYLE_SUFFIX`) before it ever reaches the model.

```bash
# one-time install (not in requirements.txt by default — heavy deps):
pip install diffusers accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # RTX 4070: CUDA build
# or, CPU-only fallback (works everywhere, much slower):
pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m clea imagine --topic "why flossing matters" --type dental \
    --format informational --style informational --music-id <id from 'clea music list'>
```

- Default model is `stabilityai/sdxl-turbo` (config `imagegen.model`) — fits
  8GB VRAM, 1-4 step inference, no CFG pass needed, fast on your RTX 4070.
  First run downloads the weights from Hugging Face (one-time, like `ollama
  pull`); after that it's fully local.
- The LLM content pack drives everything: its hook becomes the opening
  title, its script/overlay lines become the image prompts and the on-screen
  note-card text for each scene (scene order is locked chronological so the
  captions always match what's on screen — no hook-first reshuffling here).
  The dental compliance guardrail applies to the script exactly as in Phase 4.
  Cool-edit's `--format` naturally gives short on-screen-text scenes; other
  formats use the hook/problem/solution/cta as four scenes.
- Same style presets apply to pacing/transitions (`--style cool-edit` for a
  fast montage of generated art, etc.) — `keep-voice`/whisper captions are
  forced off since there's no real speech to transcribe.
- Web UI: **AI Video** tab — topic, dental/educational type, reel format,
  style, a track from the music library, length slider, generate, preview,
  download. The header's "AI images ready/not installed" chip tells you at
  a glance whether the optional deps are present.
- GPU stages (script LLM call → image generation → NVENC render) run
  strictly sequentially, same rule as everywhere else in this app — the
  pipeline never has two GPU workloads active at once.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

66 tests: beat-grid/slot invariants (cuts land on beats, pacing bounds),
EDL properties (hook-first ordering, chronological mode, no source reuse,
crossfade timeline preservation), ASS caption/note/hook generation, ffmpeg
command construction (NVENC vs CPU args, xfade offsets, keep-voice mix,
punch-in zoompan), compliance checker, style resolution, music library
scan/tag/persistence, image-prompt safety guardrail, Ken Burns clip
rendering (verified with ffprobe), the imagine pipeline end-to-end with a
mocked image generator (so it runs without torch/diffusers or a GPU), and
end-to-end renders of every style preset. Run them after any change to the
edit engine.

`tests/test_imagegen_guardrail.py` and `tests/test_imagine.py` don't need
diffusers/torch installed — the real diffusers pipeline was validated
separately against small real models (`segmind/tiny-sd`) during development
to confirm the API contract in `clea/imagegen.py` is correct; that isn't
part of the regression suite since pulling model weights on every test run
would be slow and heavy. If you change `imagegen.py`'s pipeline-call code,
sanity-check it against a real model by hand once.

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
