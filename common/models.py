"""The frozen models compared in the scaling experiment (single source of truth).

Each entry pins the Hugging Face revision, the Modal server class that serves it (all in
agent/serve_vllm.py, app "kwtc-vllm"), the GPU, and the cache directory names used locally
(art/<cache>) and on the Volume (/art/<cache>).  Gemma keeps the original names so its
existing cache stays valid.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

VLLM_APP = "kwtc-vllm"
SERVED_NAME = "llm"
ROUTING_REGION = "us-east"

MODELS: dict[str, dict] = {
    "gemma26b": {
        "label": "Gemma 4 26B-A4B", "family": "Gemma 4", "hf_id": "google/gemma-4-26B-A4B-it",
        "revision": "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301",
        "params_total": 25.8e9, "params_active": 4.0e9,     # mixture of experts: ~4B parameters active per token
        "server_cls": "Server", "gpu": "H200", "max_model_len": None,
        "cache": "cache", "cache_shortcut": "cache_shortcut",
    },
    "llama8b": {
        "label": "Llama 3.1 8B", "family": "Llama 3", "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "params_total": 8.03e9, "params_active": 8.03e9,
        "server_cls": "ServerLlama8b", "gpu": "L40S", "max_model_len": 16384,
        "cache": "cache_llama8b", "cache_shortcut": "cache_shortcut_llama8b",
    },
    "llama3b": {
        "label": "Llama 3.2 3B", "family": "Llama 3", "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "params_total": 3.21e9, "params_active": 3.21e9,
        "server_cls": "ServerLlama3b", "gpu": "L4", "max_model_len": 16384,
        "cache": "cache_llama3b", "cache_shortcut": "cache_shortcut_llama3b",
    },
}
ORDER = ["llama3b", "llama8b", "gemma26b"]       # by total parameter count


def get(key: str) -> dict:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from {list(MODELS)}")
    return {"key": key, **MODELS[key]}


def _workspace() -> str:
    ws = os.environ.get("KWTC_MODAL_WORKSPACE")
    if ws:
        return ws
    try:
        out = subprocess.run([sys.executable, "-m", "modal", "token", "info"], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"Workspace:\s*(\S+)", out)
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError("cannot determine the Modal workspace; set KWTC_MODAL_WORKSPACE")


def server_url(key: str) -> str:
    """URL of the deployed vLLM server for a model (override with KWTC_URL_<KEY>)."""
    override = os.environ.get(f"KWTC_URL_{key.upper()}")
    if override:
        return override.rstrip("/")
    m = get(key)
    return f"https://{_workspace()}--{VLLM_APP}-{m['server_cls'].lower()}.{ROUTING_REGION}.modal.direct"
