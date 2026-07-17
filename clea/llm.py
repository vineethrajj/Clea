"""LLM access: local Ollama by default, optional cloud provider as a toggle.

Everything is stdlib urllib — no extra dependency. The app must work fully
without any cloud key; cloud is only used when explicitly configured.

GPU policy: callers are responsible for never running LLM inference
concurrently with Whisper or a render — in the CLI and the web server all
GPU-capable stages run under a single sequential pipeline/lock.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Config


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM request to {url} failed: {exc}") from exc


def ollama_available(cfg: Config) -> bool:
    try:
        with urllib.request.urlopen(f"{cfg.llm['ollama_url']}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def _ollama_generate(prompt: str, system: str, cfg: Config, json_mode: bool) -> str:
    payload = {
        "model": cfg.llm["model"],
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    if json_mode:
        payload["format"] = "json"
    data = _post_json(f"{cfg.llm['ollama_url']}/api/generate", payload)
    return data.get("response", "")


def _cloud_generate(prompt: str, system: str, cfg: Config) -> str:
    provider = cfg.llm["cloud_provider"]
    key = cfg.llm["cloud_api_key"]
    if not provider or not key:
        raise LLMError("cloud provider not configured")
    if provider == "anthropic":
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {"model": "claude-sonnet-5", "max_tokens": 1500, "system": system,
             "messages": [{"role": "user", "content": prompt}]},
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in data.get("content", []))
    if provider == "groq":
        data = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": "llama-3.3-70b-versatile",
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {key}"},
        )
        return data["choices"][0]["message"]["content"]
    if provider == "gemini":
        data = _post_json(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={key}",
            {"system_instruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": prompt}]}]},
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]
    raise LLMError(f"unknown cloud provider: {provider}")


def generate(prompt: str, system: str = "", cfg: Config | None = None,
             json_mode: bool = False) -> str:
    """Route to the configured cloud provider if set, else local Ollama."""
    assert cfg is not None
    if cfg.llm["cloud_provider"] and cfg.llm["cloud_api_key"]:
        return _cloud_generate(prompt, system, cfg)
    return _ollama_generate(prompt, system, cfg, json_mode)


def parse_json_response(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose/fences — dig it out."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise
