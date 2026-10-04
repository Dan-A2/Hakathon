"""Frozen LLM: Gemma 4 26B-A4B-it served by vLLM on an H200, weights pinned by revision hash.

    modal deploy agent/serve_vllm.py
    modal run agent/serve_vllm.py          # health check + one chat completion

This follows Modal's vLLM example; only the app name, --max-logprobs and the HF secret differ.  Then
    modal secret create kwtc-llm KWTC_LLM_BACKEND=vllm KWTC_VLLM_URL=<printed URL>
"""
from __future__ import annotations

import json
import os

import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "1"})
)

MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"
SPECULATIVE_MODEL_NAME = "google/gemma-4-26B-A4B-it-assistant"
SPECULATIVE_MODEL_REVISION = "f188f476dc11dd5bb3014dc861529d316bce49d3"
SERVED_NAME = "llm"
MAX_LOGPROBS = 20          # the tok_prob feature reads the verdict token's log-prob

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Gemma weights are gated: accept the licence on huggingface.co, then
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("kwtc-vllm")

FAST_BOOT = False
N_GPU = 1
# The bf16 weights (~52 GB) need an 80 GB-class card: H200 (default), H100 or A100-80GB.
#   KWTC_VLLM_GPU=H100 modal deploy agent/serve_vllm.py
GPU_TYPE = os.environ.get("KWTC_VLLM_GPU", "H200")
MINUTES = 60
VLLM_PORT = 8000


@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    port=VLLM_PORT,
    target_concurrency=100,
    unauthenticated=True,
    secrets=[hf_secret],
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", MODEL_NAME, SERVED_NAME,
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--uvicorn-log-level=info", "--async-scheduling",
            "--max-logprobs", str(MAX_LOGPROBS),
        ]
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0}),
                "--enable-auto-tool-choice", "--reasoning-parser", "gemma4", "--tool-call-parser", "gemma4"]
        cmd += ["--speculative-config", json.dumps({"model": SPECULATIVE_MODEL_NAME, "revision": SPECULATIVE_MODEL_REVISION,
                                                    "num_speculative_tokens": 4})]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def test(test_timeout: int = 15 * MINUTES):
    """Wait for /health, then send one JSON-constrained chat completion with log-probs."""
    import time
    import urllib.request

    from openai import OpenAI

    url = Server.get_url()
    print(f"server URL: {url}  (set KWTC_VLLM_URL to this)")
    deadline = time.time() + test_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=60) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
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
