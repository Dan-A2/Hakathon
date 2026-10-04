"""Versioned prompt files. They are hashed into controller.npz and must not change after the cache is built."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common.config import PROMPTS_DIR
from common.io import sha256_text

PROMPT_FILES = ("provisional.txt", "verify.txt")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, str]:
    """Parse a prompt file with '=== SYSTEM ===' and '=== USER ===' sections."""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("=== ") and line.rstrip().endswith(" ==="):
            current = line.strip("= ").strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def fill(template: str, **values) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


@lru_cache(maxsize=None)
def prompt_hash() -> str:
    parts = []
    for name in PROMPT_FILES:
        parts.append(name + "\n" + (PROMPTS_DIR / name).read_text(encoding="utf-8"))
    return sha256_text("\n".join(parts))[:16]


def format_evidence(initial_evidence: list[dict]) -> str:
    if not initial_evidence:
        return "(no abstracts were retrieved)"
    blocks = []
    for ev in initial_evidence:
        blocks.append(f"[doc_id {ev['doc_id']}] {ev.get('title', '')}\n{ev.get('text', '')}")
    return "\n\n".join(blocks)
