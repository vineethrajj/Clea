"""Hardware capability detection: NVIDIA GPU, NVENC, CUDA-for-Whisper, Ollama.

Everything degrades gracefully: each probe returns a result object rather than
raising, so the pipeline can pick fallbacks (libx264, CPU whisper, cloud-less
LLM mode) and the UI can show what's active.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field


@dataclass
class HardwareReport:
    ffmpeg: bool = False
    ffmpeg_version: str = ""
    nvidia_gpu: bool = False
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    nvenc: bool = False
    cuda_whisper: bool = False
    ollama: bool = False
    ollama_models: list[str] = field(default_factory=list)

    @property
    def encoder(self) -> str:
        return "h264_nvenc" if self.nvenc else "libx264"


def _run(cmd: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_ffmpeg() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, ""
    proc = _run(["ffmpeg", "-version"])
    if proc is None or proc.returncode != 0:
        return False, ""
    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    return True, first_line.replace("ffmpeg version ", "").split(" ")[0]


def detect_nvidia_gpu() -> tuple[bool, str, int]:
    proc = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"])
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return False, "", 0
    name, _, mem = proc.stdout.strip().splitlines()[0].partition(",")
    try:
        vram = int(mem.strip())
    except ValueError:
        vram = 0
    return True, name.strip(), vram


def detect_nvenc() -> bool:
    """Actually test-encode one frame — the encoder can be compiled into
    ffmpeg while no usable NVIDIA device is present, so listing encoders
    is not enough."""
    proc = _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
    ], timeout=30.0)
    return proc is not None and proc.returncode == 0


def detect_cuda_whisper() -> bool:
    """CUDA availability for faster-whisper (via ctranslate2)."""
    try:
        import ctranslate2  # noqa: PLC0415 — optional, Phase 2 dependency
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def detect_ollama(base_url: str = "http://localhost:11434") -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            data = json.load(resp)
        return True, [m.get("name", "?") for m in data.get("models", [])]
    except Exception:
        return False, []


def probe(ollama_url: str = "http://localhost:11434") -> HardwareReport:
    report = HardwareReport()
    report.ffmpeg, report.ffmpeg_version = detect_ffmpeg()
    report.nvidia_gpu, report.gpu_name, report.gpu_vram_mb = detect_nvidia_gpu()
    if report.ffmpeg:
        report.nvenc = detect_nvenc()
    report.cuda_whisper = detect_cuda_whisper()
    report.ollama, report.ollama_models = detect_ollama(ollama_url)
    return report


def format_report(r: HardwareReport) -> str:
    def mark(ok: bool) -> str:
        return "[OK]  " if ok else "[--]  "

    lines = [
        "Clea hardware check",
        "-" * 46,
        f"{mark(r.ffmpeg)}ffmpeg          {r.ffmpeg_version or 'NOT FOUND (required)'}",
        f"{mark(r.nvidia_gpu)}NVIDIA GPU      "
        + (f"{r.gpu_name} ({r.gpu_vram_mb} MB VRAM)" if r.nvidia_gpu else "not detected"),
        f"{mark(r.nvenc)}NVENC encode    "
        + ("h264_nvenc active" if r.nvenc else "falling back to libx264 (CPU)"),
        f"{mark(r.cuda_whisper)}Whisper CUDA    "
        + ("GPU transcription available" if r.cuda_whisper
           else "CPU fallback (small model) — or faster-whisper not installed yet"),
        f"{mark(r.ollama)}Ollama          "
        + (f"running, models: {', '.join(r.ollama_models) or '(none pulled)'}"
           if r.ollama else "not running — AI content generator disabled"),
    ]
    return "\n".join(lines)
