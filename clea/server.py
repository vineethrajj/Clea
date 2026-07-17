"""Phase 5: local web server — FastAPI + single-page frontend.

Runs on your machine; reachable from a phone on the same wifi via the PC's
LAN IP (printed at startup). No cloud, no accounts.

Concurrency model (8GB VRAM rule): one global GPU_LOCK serialises every
GPU-capable operation — edit pipeline runs (whisper + render) and LLM
generations. Jobs queue behind it in submission order; nothing GPU-related
ever runs concurrently.
"""

from __future__ import annotations

import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .config import Config, load_config
from .content import CONTENT_TYPES, generate_content
from .hardware import HardwareReport, detect_ollama, probe
from .llm import LLMError, ollama_available
from .pipeline import EditOptions, collect_clips, run_edit

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Clea")

GPU_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
_cfg: Config | None = None
_hw: HardwareReport | None = None


def _workspace() -> Path:
    assert _cfg is not None
    root = Path(_cfg.paths["output_dir"]) / "webjobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


@app.on_event("startup")
def _startup() -> None:
    global _cfg, _hw
    if _cfg is None:
        _cfg = load_config()
    _hw = probe(_cfg.llm["ollama_url"])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    assert _cfg is not None and _hw is not None
    # Ollama can start/stop while the server runs — re-check it live.
    ollama_up, models = detect_ollama(_cfg.llm["ollama_url"])
    return {
        "ffmpeg": _hw.ffmpeg,
        "nvenc": _hw.nvenc,
        "gpu_name": _hw.gpu_name,
        "cuda_whisper": _hw.cuda_whisper,
        "ollama": ollama_up,
        "ollama_models": models,
        "llm_model": _cfg.llm["model"],
        "encoder": _hw.encoder,
        "busy": GPU_LOCK.locked(),
    }


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    try:
        job["state"] = "waiting for GPU"
        with GPU_LOCK:  # one GPU workload at a time, ever
            def progress(stage: str, msg: str) -> None:
                job["state"] = stage
                job["message"] = msg

            assert _cfg is not None
            result = run_edit(
                job["clips"], job["audio"], job["out"], _cfg,
                job["options"], progress=progress, hw=_hw,
            )
        job.update(state="done", message="ready",
                   encoder=result.encoder,
                   video_duration=result.total_duration,
                   n_cuts=result.n_cuts,
                   n_caption_words=result.n_caption_words,
                   notes=result.note_texts,
                   render_seconds=round(result.render_seconds, 1))
    except Exception as exc:
        job.update(state="error", message=str(exc)[-2000:])


@app.post("/api/edit")
async def create_edit(
    clips: list[UploadFile] = File(...),
    audio: UploadFile = File(...),
    duration: float = Form(20.0),
    style: str = Form("informational"),
    captions: bool = Form(False),
    notes: bool = Form(False),
    keep_voice: bool = Form(False),
    no_xfade: bool = Form(False),
    hook_text: str = Form(""),
) -> dict:
    assert _cfg is not None
    if style not in (_cfg.get("styles") or {}):
        raise HTTPException(400, f"unknown style '{style}'")
    job_id = uuid.uuid4().hex[:12]
    ws = _workspace() / job_id
    clips_dir = ws / "clips"
    clips_dir.mkdir(parents=True)

    for i, up in enumerate(clips):
        suffix = Path(up.filename or f"clip{i}.mp4").suffix or ".mp4"
        with open(clips_dir / f"{i:03d}{suffix}", "wb") as fh:
            shutil.copyfileobj(up.file, fh)
    audio_suffix = Path(audio.filename or "track.mp3").suffix or ".mp3"
    audio_path = ws / f"audio{audio_suffix}"
    with open(audio_path, "wb") as fh:
        shutil.copyfileobj(audio.file, fh)

    clip_paths = collect_clips(clips_dir)
    if not clip_paths:
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(400, "no usable video clips in upload")

    JOBS[job_id] = {
        "id": job_id, "state": "queued", "message": "queued",
        "created": time.time(),
        "clips": clip_paths, "audio": str(audio_path),
        "out": str(ws / "reel.mp4"),
        "options": EditOptions(duration=duration, style=style,
                               captions=captions, notes=notes,
                               keep_voice=keep_voice, no_xfade=no_xfade,
                               hook_text=hook_text.strip() or None),
    }
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    public = {k: v for k, v in job.items()
              if k in ("id", "state", "message", "encoder", "video_duration",
                       "n_cuts", "n_caption_words", "notes", "render_seconds")}
    return public


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["state"] != "done":
        raise HTTPException(409, f"job is {job['state']}")
    return FileResponse(job["out"], media_type="video/mp4",
                        filename="clea_reel.mp4")


class GenerateRequest(BaseModel):
    topic: str
    content_type: str = "educational-other"
    reel_format: str = "informational"


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    assert _cfg is not None
    if req.content_type not in CONTENT_TYPES:
        raise HTTPException(400, f"content_type must be one of {CONTENT_TYPES}")
    if not req.topic.strip():
        raise HTTPException(400, "topic is required")
    uses_cloud = _cfg.llm["cloud_provider"] and _cfg.llm["cloud_api_key"]
    if not uses_cloud and not ollama_available(_cfg):
        raise HTTPException(
            503, f"Ollama is not running. Start it and pull {_cfg.llm['model']}.")
    try:
        with GPU_LOCK:  # LLM inference is a GPU stage too
            pack = generate_content(req.topic.strip(), req.content_type, _cfg,
                                    reel_format=req.reel_format)
    except (LLMError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return pack.to_dict()


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no traffic sent; just picks the route
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def serve(host: str = "0.0.0.0", port: int = 8000,
          config_path: str | None = None) -> None:
    import uvicorn

    global _cfg
    _cfg = load_config(config_path)
    print(f"\n  Clea web UI:\n"
          f"    this machine:  http://localhost:{port}\n"
          f"    on your phone: http://{lan_ip()}:{port}  (same wifi)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
