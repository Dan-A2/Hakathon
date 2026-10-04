"""Frozen LLMs served by vLLM on Modal, weights pinned by revision hash (one server class per model).

    modal deploy agent/serve_vllm.py                         # registers all servers; GPUs start only on demand
    modal run agent/serve_vllm.py --model llama8b            # health check + one JSON completion with log-probs

Models, revisions and GPUs come from common/models.py.  Gemma follows Modal's vLLM example
(speculative decoding, Gemma parsers); the Llama servers use plain `vllm serve`.  Llama weights
are gated: request access on huggingface.co for the account whose token is in `huggingface-secret`.
"""
from __future__ import annotations

import json

import modal

from common.models import MODELS, SERVED_NAME, VLLM_APP

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # Gemma 4 needs Transformers v5 (v4 does not know model type "gemma4"), but Transformers >= 5.15 makes
    # per-layer config attributes like head_dim raise on global access, which vllm 0.21.0 does. 5.8.1 is the
    # release current when vllm 0.21.0 shipped (2026-05-15) and has Gemma 4 without that change.
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "1"})
    .add_local_python_source("common")
)

GEMMA_SPECULATIVE = {"model": "google/gemma-4-26B-A4B-it-assistant",
                     "revision": "f188f476dc11dd5bb3014dc861529d316bce49d3", "num_speculative_tokens": 4}
MAX_LOGPROBS = 20          # the tok_prob feature reads the verdict token's log-prob
MINUTES = 60
VLLM_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")    # modal secret create huggingface-secret HF_TOKEN=hf_...

app = modal.App(VLLM_APP)


def _server_kwargs(key: str) -> dict:
    return dict(
        image=vllm_image, gpu=f"{MODELS[key]['gpu']}:1", scaledown_window=15 * MINUTES, startup_timeout=10 * MINUTES,
        volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
        port=VLLM_PORT, target_concurrency=100, unauthenticated=True, secrets=[hf_secret],
    )


def _launch(key: str):
    """Start `vllm serve` for a model and a watchdog that stops the container if vLLM dies."""
    import os
    import subprocess
    import threading

    m = MODELS[key]
    cmd = ["vllm", "serve", m["hf_id"], "--revision", m["revision"],
           "--served-model-name", m["hf_id"], SERVED_NAME,
           "--host", "0.0.0.0", "--port", str(VLLM_PORT), "--uvicorn-log-level=info", "--async-scheduling",
           "--max-logprobs", str(MAX_LOGPROBS), "--no-enforce-eager", "--tensor-parallel-size", "1"]
    if m["max_model_len"]:
        cmd += ["--max-model-len", str(m["max_model_len"])]
    if key == "gemma26b":
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0}),
                "--enable-auto-tool-choice", "--reasoning-parser", "gemma4", "--tool-call-parser", "gemma4",
                "--speculative-config", json.dumps(GEMMA_SPECULATIVE)]
    print(*cmd, flush=True)
    proc = subprocess.Popen(cmd)

    def _watch():
        code = proc.wait()
        print(f"vllm serve exited with code {code}; stopping the container", flush=True)
        os._exit(code if code is not None else 1)

    threading.Thread(target=_watch, daemon=True).start()
    return proc


def _stop(proc):
    proc.terminate()
    proc.wait(timeout=30)


@app.server(**_server_kwargs("gemma26b"))
class Server:                       # name kept so Gemma's URL stays the same
    @modal.enter()
    def start(self):
        self.process = _launch("gemma26b")

    @modal.exit()
    def stop(self):
        _stop(self.process)


@app.server(**_server_kwargs("llama8b"))
class ServerLlama8b:
    @modal.enter()
    def start(self):
        self.process = _launch("llama8b")

    @modal.exit()
    def stop(self):
        _stop(self.process)


@app.server(**_server_kwargs("llama3b"))
class ServerLlama3b:
    @modal.enter()
    def start(self):
        self.process = _launch("llama3b")

    @modal.exit()
    def stop(self):
        _stop(self.process)


SERVERS = {"gemma26b": Server, "llama8b": ServerLlama8b, "llama3b": ServerLlama3b}


@app.local_entrypoint()
def test(model: str = "gemma26b", test_timeout: int = 15 * MINUTES):
    """Wait for /health, then send one JSON-constrained chat completion with log-probs."""
    import time
    import urllib.request

    from openai import OpenAI

    url = SERVERS[model].get_url()
    print(f"{model} server URL: {url}")
    deadline = time.time() + test_timeout
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=60) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            pass
        if time.time() > deadline:
            raise SystemExit(f"{model} server never became healthy within {test_timeout}s; check the container logs")
        time.sleep(5)
    client = OpenAI(base_url=url + "/v1", api_key="EMPTY")
    resp = client.chat.completions.create(
        model=SERVED_NAME, temperature=0.7, seed=0, max_tokens=120, logprobs=True, top_logprobs=1,
        messages=[{"role": "user", "content": 'Reply with JSON only: {"verdict": "supported" | "refuted", "confidence": 0-1}. '
                                              "Claim: water boils at 100 C at sea level."}],
        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": {
            "type": "object", "properties": {"verdict": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["verdict", "confidence"], "additionalProperties": False}}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(resp.choices[0].message.content)
    print("tokens with log-probs:", len(resp.choices[0].logprobs.content or []))
