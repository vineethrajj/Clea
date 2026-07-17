"""Phase 4: AI content generator — topic in, short-form content pack out.

Runs on local Ollama (qwen2.5:7b) by default; the optional cloud toggle in
config.yaml routes through llm.generate() transparently.

Content types:
  * dental            — compliance guardrail enforced in the system prompt:
                        no absolute/guarantee claims, no unverifiable
                        outcomes, soft framing, "consult your dentist"
  * educational-other — no compliance constraint, just clear + engaging
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import llm
from .config import Config

CONTENT_TYPES = ("dental", "educational-other")

_BASE_SYSTEM = """You are a short-form video content strategist writing for
Instagram Reels. You write punchy, concrete, scroll-stopping copy in simple
language. Never use emojis in the script itself. Always respond with ONLY a
JSON object, no prose around it."""

_DENTAL_GUARDRAIL = """
COMPLIANCE RULES (dental marketing — these override everything else):
- NEVER use absolute or guarantee words: "instant", "instantly", "painless",
  "pain-free", "best", "cure", "cures", "guaranteed", "100%", "permanent",
  "always works", "no side effects".
- NEVER promise specific outcomes or timeframes for results.
- Prefer soft, accurate framing: "gently", "in most cases", "can help",
  "may reduce", "designed to be comfortable".
- Where a claim depends on the patient, add "consult your dentist" or
  "your dentist can advise what's right for you".
- Educational tone over sales tone."""

_EDU_SYSTEM = """
Content style: clear, accurate, engaging explanation for learners (exam
prep / concept education). Use a concrete example or memorable framing in
the explanation. No compliance restrictions, but never invent facts."""

_PROMPT_TEMPLATE = """Topic: {topic}

Create a short-form video content pack. Return ONLY this JSON shape:
{{
  "hook": "one scroll-stopping opening line, under 12 words",
  "script": {{
    "hook": "spoken opening line(s), 1-2 sentences",
    "problem": "why the viewer should care / what goes wrong, 1-2 sentences",
    "solution": "the explanation or answer, 2-3 sentences",
    "cta": "one closing call-to-action sentence"
  }},
  "caption": "an Instagram caption, 1-2 sentences plus a question to drive comments",
  "hashtags": ["5 to 8 relevant hashtags, each starting with #"]
}}"""

# Compliance words checked *after* generation as a safety net; if the model
# slips one in, we regenerate once, then flag it in the response.
_BANNED_DENTAL = (
    "instant", "painless", "pain-free", "pain free", "guaranteed", "100%",
    "cure", "permanent", "best ", "no side effects", "always works",
)


@dataclass
class ContentPack:
    topic: str
    content_type: str
    hook: str = ""
    script: dict = field(default_factory=dict)
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    compliance_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _system_prompt(content_type: str) -> str:
    if content_type == "dental":
        return _BASE_SYSTEM + _DENTAL_GUARDRAIL
    return _BASE_SYSTEM + _EDU_SYSTEM


def _compliance_check(pack: ContentPack) -> list[str]:
    blob = " ".join([
        pack.hook, pack.caption,
        " ".join(str(v) for v in pack.script.values()),
    ]).lower()
    return [w for w in _BANNED_DENTAL if w in blob]


def _generate_once(topic: str, content_type: str, cfg: Config) -> ContentPack:
    raw = llm.generate(
        _PROMPT_TEMPLATE.format(topic=topic),
        system=_system_prompt(content_type),
        cfg=cfg,
        json_mode=True,
    )
    data = llm.parse_json_response(raw)
    script = data.get("script") or {}
    if not isinstance(script, dict):
        script = {"hook": str(script)}
    hashtags = [str(h) if str(h).startswith("#") else f"#{h}"
                for h in data.get("hashtags", [])][:8]
    return ContentPack(
        topic=topic,
        content_type=content_type,
        hook=str(data.get("hook", "")).strip(),
        script={k: str(v).strip() for k, v in script.items()},
        caption=str(data.get("caption", "")).strip(),
        hashtags=hashtags,
    )


def generate_content(topic: str, content_type: str, cfg: Config) -> ContentPack:
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"content_type must be one of {CONTENT_TYPES}")
    pack = _generate_once(topic, content_type, cfg)
    if content_type == "dental":
        violations = _compliance_check(pack)
        if violations:  # one retry with explicit feedback, then flag
            pack = _generate_once(
                topic + "\n\nIMPORTANT: your previous draft used banned words "
                f"({', '.join(violations)}). Do not use them.", content_type, cfg)
            pack.topic = topic
            pack.compliance_warnings = _compliance_check(pack)
    return pack


def format_pack(pack: ContentPack) -> str:
    lines = [
        f"TOPIC: {pack.topic}   [{pack.content_type}]",
        "",
        f"HOOK: {pack.hook}",
        "",
        "SCRIPT",
    ]
    for part in ("hook", "problem", "solution", "cta"):
        if part in pack.script:
            lines.append(f"  {part.upper():9s} {pack.script[part]}")
    lines += ["", f"CAPTION: {pack.caption}", "", "HASHTAGS: " + " ".join(pack.hashtags)]
    if pack.compliance_warnings:
        lines += ["", "!! COMPLIANCE WARNINGS (review before posting): "
                  + ", ".join(pack.compliance_warnings)]
    return "\n".join(lines)
