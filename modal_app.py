"""Modal integration: every stage is one Modal primitive.

    modal run modal_app.py::upload_cases                 # push data/cases to the Volume (once)
    modal run modal_app.py::cache --split train          # counterfactual cache via .map()
    modal run modal_app.py::sweep                        # 4 w x 5 c x 5 seeds via .starmap()
    modal run modal_app.py::upload_artifacts             # push controller.npz / calibrator.npz for the demo
    modal deploy modal_app.py                            # live demo URL (FastAPI, @modal.asgi_app)

The LLM credentials live in a Modal Secret named "kwtc-llm", e.g.
    modal secret create kwtc-llm KWTC_LLM_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-...
or, for the vLLM server from agent/serve_vllm.py,
    modal secret create kwtc-llm KWTC_LLM_BACKEND=vllm KWTC_VLLM_URL=https://<workspace>--kwtc-vllm-server.modal.run
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "kwtc"
ART = "/art"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("kwtc-artifacts", create_if_missing=True)
llm_secret = modal.Secret.from_name("kwtc-llm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "rank_bm25", "openai", "anthropic", "fastapi[standard]", "matplotlib")
    .env({"KWTC_ART_DIR": ART, "KWTC_CASES_DIR": f"{ART}/data/cases", "KWTC_CACHE_DIR": f"{ART}/cache",
          "KWTC_CALC_BACKEND": "modal"})
    .add_local_python_source("common", "data", "agent", "controller", "scorer", "eval", "demo", "modal_app")
    .add_local_dir("agent/prompts", remote_path="/root/agent/prompts", ignore=lambda p: p.suffix == ".py")
    .add_local_file("demo/index.html", remote_path="/root/demo/index.html")
)

_CASES: dict[str, dict[str, dict]] = {}   # per-container cache of split -> case_id -> case


def _cases(split: str) -> dict[str, dict]:
    if split not in _CASES:
        from common.io import read_jsonl

        _CASES[split] = {r["case_id"]: r for r in read_jsonl(f"{ART}/data/cases/{split}.jsonl")}
    return _CASES[split]


# ----------------------------------------------------------------------------- 2. counterfactual cache
@app.function(image=image, volumes={ART: vol}, secrets=[llm_secret], timeout=900, retries=2, max_containers=64)
def build_case(case_id: str, split: str, K: int = 4) -> dict:
    """Provisional x2 + verify for one case. Idempotent: an existing record on the Volume is returned as is."""
    from agent.build_cache import build_case as _build, case_cache_path
    from agent.llm import get_llm
    from common.io import load_json, save_json
    from data.records import StoreRegistry

    out = case_cache_path(Path(f"{ART}/cache"), split, case_id)
    if out.exists():
        return load_json(out)
    case = _cases(split)[case_id]
    rec = _build(case, get_llm(), StoreRegistry(f"{ART}/data/cases"), K=K, calc_backend="modal")
    save_json(out, rec, indent=None)
    vol.commit()
    return rec


@app.function(image=image, volumes={ART: vol}, timeout=1800)
def merge(split: str) -> str:
    from agent.build_cache import merge_split

    vol.reload()
    out = merge_split(Path(f"{ART}/cache"), split)
    vol.commit()
    return str(out)


# ----------------------------------------------------------------------------- 3. sandboxed calculator
@app.function(image=image, timeout=120)
def run_calc(code: str) -> str:
    """LLM-written arithmetic never runs on our machines: a Sandbox with no network and a 10 s timeout."""
    sb = modal.Sandbox.create(app=app, image=modal.Image.debian_slim(python_version="3.12"),
                              block_network=True, timeout=10)
    try:
        p = sb.exec("python", "-c", code, timeout=10)
        out = p.stdout.read()
        err = p.stderr.read()
    finally:
        sb.terminate()
    return out if out.strip() else f"error: {err.strip()[:300]}"


# ----------------------------------------------------------------------------- 4. reward sweep
@app.function(image=image, volumes={ART: vol}, cpu=2, timeout=1800)
def train_one(w: float, c: float, seed: int, use_tok_prob: bool = False) -> dict:
    from controller.train import train_one as _train_one

    vol.reload()
    return _train_one(w, c, seed, cases_dir=f"{ART}/data/cases", cache_dir=f"{ART}/cache", use_tok_prob=use_tok_prob)


# ----------------------------------------------------------------------------- 5. live demo
@app.function(image=image, volumes={ART: vol}, secrets=[llm_secret], timeout=600, scaledown_window=600)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from demo.serve import create_app

    return create_app(controller=f"{ART}/controller.npz", calibrator=f"{ART}/calibrator.npz",
                      cases_dir=f"{ART}/data/cases", cache_dir=f"{ART}/cache")


# ----------------------------------------------------------------------------- local entrypoints
def _download(remote: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(vol.read_file(remote))
    local.write_bytes(data)
    return local


@app.local_entrypoint()
def upload_cases(cases_dir: str = "data/cases"):
    with vol.batch_upload(force=True) as b:
        b.put_directory(cases_dir, "/data/cases")
    print(f"uploaded {cases_dir} -> kwtc-artifacts:/data/cases")


@app.local_entrypoint()
def upload_artifacts(controller: str = "art/controller.npz", calibrator: str = "art/calibrator.npz"):
    with vol.batch_upload(force=True) as b:
        b.put_file(controller, "/controller.npz")
        if Path(calibrator).exists():
            b.put_file(calibrator, "/calibrator.npz")
    print("uploaded controller (+calibrator) to kwtc-artifacts:/")


@app.local_entrypoint()
def cache(split: str = "train", k: int = 4, limit: int = 0, cases_dir: str = "data/cases", out_dir: str = "art/cache"):
    """Build the cache for one split with .map(), merge on the Volume, download the jsonl."""
    from common.io import read_jsonl

    ids = [r["case_id"] for r in read_jsonl(f"{cases_dir}/{split}.jsonl")]
    if limit:
        ids = ids[:limit]
    n_ok = n_err = 0
    for res in build_case.map(ids, kwargs={"split": split, "K": k}, return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            n_err += 1
            print(f"  error: {res!r}")
        else:
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  {n_ok}/{len(ids)} cases cached")
    remote = merge.remote(split)
    local = _download(f"cache/{split}.jsonl", Path(out_dir) / f"{split}.jsonl")
    print(f"{split}: {n_ok} ok, {n_err} errors; merged {remote}; downloaded to {local}")


@app.local_entrypoint()
def sweep(seeds: int = 5, out: str = "art/sweep.jsonl", ws: str = "0.5,1,2,4", cs: str = "0,0.025,0.05,0.1,0.2"):
    """100 CPU runs in parallel -> the phase diagram and the cost-accuracy frontier."""
    grid = [(float(w), float(c), s) for w in ws.split(",") for c in cs.split(",") for s in range(seeds)]
    rows = list(train_one.starmap(grid))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} sweep rows to {out}")


@app.local_entrypoint()
def download(remote: str, local: str):
    print(_download(remote, Path(local)))


@app.local_entrypoint()
def calc(expression: str = "2 ** 10 + 1"):
    """Smoke-test the sandboxed calculator."""
    print(run_calc.remote(f"print({expression})"))
