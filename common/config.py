"""Project-wide constants, paths and environment-driven settings.

Nothing in here reads gold labels. Paths can be redirected with environment
variables so the same code runs locally and inside Modal containers (where the
artifacts live on a Volume mounted at /art).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- actions / labels / features -------------------------------------------------
ACTIONS = ["answer", "verify", "abstain"]          # indices 0, 1, 2
ANSWER, VERIFY, ABSTAIN = 0, 1, 2
LABELS = ["supported", "refuted", "insufficient_evidence"]
COMMIT_LABELS = ("supported", "refuted")           # "S/R" verdicts

# The six pre-action signals (bias first). tok_prob is optional (vLLM only).
FEATURE_NAMES = ["bias", "conf_mean", "agree", "suff_mean", "conf_gap", "says_insuff"]
TOK_PROB_FEATURE = "tok_prob"

# ---- reward knobs ---------------------------------------------------------------------
DEFAULT_W = 1.0        # wrong-answer penalty
DEFAULT_C = 0.05       # tool cost per call
DEFAULT_K = 4          # tool-call limit in the verification loop

# ---- sampling ------------------------------------------------------------------------------
PROVISIONAL_TEMPERATURE = 0.7
VERIFY_TEMPERATURE = 0.3
INITIAL_EVIDENCE_K = 3
SEARCH_K = 5
ABSTRACT_MAX_WORDS = 250

# ---- paths (env-overridable) ----------------------------------------------------------
def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


DATA_DIR = ROOT / "data"
RAW_DIR = _env_path("KWTC_RAW_DIR", DATA_DIR / "raw")
CASES_DIR = _env_path("KWTC_CASES_DIR", DATA_DIR / "cases")
ART_DIR = _env_path("KWTC_ART_DIR", ROOT / "art")
CACHE_DIR = _env_path("KWTC_CACHE_DIR", ART_DIR / "cache")
PROMPTS_DIR = ROOT / "agent" / "prompts"

SPLITS = ["train", "val", "test_id", "test_ood"]

# ---- LLM backend -------------------------------------------------------------------------
# KWTC_LLM_BACKEND: "vllm" | "claude" | "mock" (auto-detected when unset)
VLLM_URL = os.environ.get("KWTC_VLLM_URL", "")
VLLM_MODEL = os.environ.get("KWTC_VLLM_MODEL", "llm")          # --served-model-name alias
VLLM_MODEL_ID = os.environ.get("KWTC_VLLM_MODEL_ID", "google/gemma-4-26B-A4B-it")
VLLM_REVISION = os.environ.get("KWTC_VLLM_REVISION", "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301")
CLAUDE_MODEL = os.environ.get("KWTC_CLAUDE_MODEL", "claude-haiku-4-5")
CALC_BACKEND = os.environ.get("KWTC_CALC_BACKEND", "local")      # "local" (AST-safe) | "modal" (Sandbox)


def llm_backend() -> str:
    b = os.environ.get("KWTC_LLM_BACKEND", "").strip().lower()
    if b:
        return b
    if VLLM_URL:
        return "vllm"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    return "mock"
