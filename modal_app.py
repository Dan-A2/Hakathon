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
    .add_local_file("demo/live.html", remote_path="/root/demo/live.html")
    .add_local_file("demo/replay.js", remote_path="/root/demo/replay.js")
)

_CASES: dict[tuple[str, str], dict[str, dict]] = {}   # per-container cache of (cases_name, split) -> case_id -> case


def _cases(split: str, cases_name: str = "cases") -> dict[str, dict]:
    key = (cases_name, split)
    if key not in _CASES:
        from common.io import read_jsonl

        _CASES[key] = {r["case_id"]: r for r in read_jsonl(f"{ART}/data/{cases_name}/{split}.jsonl")}
    return _CASES[key]


# ----------------------------------------------------------------------------- 2. counterfactual cache
def _make_llm(llm_spec: dict | None):
    """The kwtc-llm secret's backend by default; an explicit vLLM server when llm_spec is given (--model)."""
    from agent.llm import VLLMClient, get_llm

    if not llm_spec:
        return get_llm()
    return VLLMClient(url=llm_spec["url"], model_id=llm_spec["model_id"], revision=llm_spec["revision"])


@app.function(image=image, volumes={ART: vol}, secrets=[llm_secret], timeout=900, retries=2, max_containers=64)
def build_case(case_id: str, split: str, K: int = 4, cases_name: str = "cases", cache_name: str = "cache",
               llm_spec: dict | None = None) -> dict:
    """Provisional x2 + verify for one case. Idempotent: an existing record on the Volume is returned as is."""
    from agent.build_cache import build_case as _build, case_cache_path
    from common.io import load_json, save_json
    from data.records import StoreRegistry

    out = case_cache_path(Path(f"{ART}/{cache_name}"), split, case_id)
    if out.exists():
        return load_json(out)
    case = _cases(split, cases_name)[case_id]
    try:
        rec = _build(case, _make_llm(llm_spec), StoreRegistry(f"{ART}/data/{cases_name}"), K=K, calc_backend="modal")
    except Exception as e:  # noqa: BLE001 - SDK exception types don't deserialize locally; send a plain message
        raise RuntimeError(f"{case_id}: {type(e).__name__}: {e}"[:600]) from None
    save_json(out, rec, indent=None)
    vol.commit()
    return rec


@app.function(image=image, volumes={ART: vol}, secrets=[llm_secret], timeout=600, retries=2, max_containers=64)
def judge_case(case_id: str, split: str, cache_name: str = "cache", cases_name: str = "cases", llm_spec: dict | None = None) -> dict:
    """Blind judge for one cached case: an evidence-only LLM call on the sentences the verifier cited."""
    from agent.build_judge import judge_record, needs_judge
    from common.io import load_json, save_json
    from data.records import StoreRegistry

    out = Path(f"{ART}/{cache_name}/judge/{split}/{case_id}.json")
    if out.exists():
        return load_json(out)
    rec = load_json(Path(f"{ART}/{cache_name}/{split}/{case_id}.json"))
    case = _cases(split, cases_name)[case_id]
    registry = StoreRegistry(f"{ART}/data/{cases_name}")
    quotes = needs_judge(rec, registry, case)
    if not quotes:
        return {"case_id": case_id, "skipped": True}
    try:
        row = judge_record(rec, case, quotes, _make_llm(llm_spec))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{case_id}: {type(e).__name__}: {e}"[:600]) from None
    save_json(out, row, indent=None)
    vol.commit()
    return row


@app.function(image=image, volumes={ART: vol}, timeout=1800)
def merge_judge(split: str, cache_name: str = "cache") -> str:
    from agent.build_judge import merge_judge as _merge

    vol.reload()
    out = _merge(Path(f"{ART}/{cache_name}"), split)
    vol.commit()
    return str(out)


@app.function(image=image, secrets=[llm_secret], timeout=900)
def check_llm(llm_spec: dict | None = None) -> dict:
    """Preflight: which backend/URL is selected (secret or --model), and one real JSON call through agent.llm."""
    import os
    import time
    import urllib.error
    import urllib.request

    from common import config as Cfg

    if llm_spec:
        info = {"backend": "vllm", "vllm_url": llm_spec["url"], "claude_model": None, "model_id": llm_spec["model_id"]}
    else:
        info = {"backend": Cfg.llm_backend(), "vllm_url": os.environ.get("KWTC_VLLM_URL", ""),
                "claude_model": Cfg.CLAUDE_MODEL if Cfg.llm_backend() == "claude" else None}
    if info["backend"] == "vllm":
        url = info["vllm_url"].rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        problems = []
        if "modal.com/apps" in url:
            problems.append("this is a Modal dashboard link, not the server URL")
        if "-dev." in url:
            problems.append("this is the temporary URL from `modal run`; use the one printed by `modal deploy`")
        info["url_problems"] = problems
        deadline, status = time.time() + 600, None          # cold start: weights load in a few minutes
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url + "/health", timeout=60) as r:
                    status = r.status
            except urllib.error.HTTPError as e:
                status = e.code
            except Exception as e:  # noqa: BLE001
                status = repr(e)[:120]
            if status == 200 or status not in (503, 502, 504) and not isinstance(status, str):
                break
            time.sleep(10)
        info["health"] = status
        if status != 200:
            return info
    try:
        res = _make_llm(llm_spec).chat([{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}],
                             temperature=0.0, max_tokens=20, want_logprobs=info["backend"] == "vllm")
        info.update({"chat_ok": True, "reply": res.text[:120], "logprobs": bool(res.logprobs), "model": res.model})
    except Exception as e:  # noqa: BLE001
        info.update({"chat_ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
    return info


def _llm_spec(model: str) -> dict | None:
    """Resolve a registry model to {url, model_id, revision}; '' means use the kwtc-llm secret."""
    if not model:
        return None
    from common.models import get, server_url

    m = get(model)
    return {"url": server_url(model), "model_id": m["hf_id"], "revision": m["revision"]}


def _preflight(model: str = "") -> None:
    info = check_llm.remote(_llm_spec(model))
    print("LLM preflight:", json.dumps(info, indent=2))
    if not info.get("chat_ok"):
        raise SystemExit("LLM preflight failed; fix the kwtc-llm secret (see above) before building the cache.")


@app.local_entrypoint()
def preflight(model: str = ""):
    _preflight(model)


@app.function(image=image, volumes={ART: vol}, timeout=1800)
def merge(split: str, cache_name: str = "cache") -> str:
    from agent.build_cache import merge_split

    vol.reload()
    out = merge_split(Path(f"{ART}/{cache_name}"), split)
    vol.commit()
    return str(out)


# ----------------------------------------------------------------------------- 3. sandboxed calculator
@app.function(image=image, timeout=120)
def run_calc(code: str) -> str:
    """LLM-written arithmetic never runs on our machines: a Sandbox with no network and a 10 s timeout."""
    # `timeout` is the sandbox's whole lifetime: it must outlive scheduling + the 10 s exec limit below
    sb = modal.Sandbox.create(app=app, image=modal.Image.debian_slim(python_version="3.12"),
                              block_network=True, timeout=60)
    try:
        p = sb.exec("python", "-c", code, timeout=10)
        out = p.stdout.read()
        err = p.stderr.read()
    finally:
        sb.terminate()
    return out if out.strip() else f"error: {err.strip()[:300]}"


# ----------------------------------------------------------------------------- 4. reward sweep
@app.function(image=image, volumes={ART: vol}, cpu=2, timeout=1800)
def train_one(w: float, c: float, seed: int, use_tok_prob: bool = False, cache_name: str = "cache") -> dict:
    from controller.train import train_one as _train_one

    vol.reload()
    return _train_one(w, c, seed, cases_dir=f"{ART}/data/cases", cache_dir=f"{ART}/{cache_name}", use_tok_prob=use_tok_prob)


# ----------------------------------------------------------------------------- 5. live demo
@app.function(image=image, volumes={ART: vol}, secrets=[llm_secret], timeout=600, scaledown_window=600)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from demo.serve import create_app

    return create_app(controller=f"{ART}/controller.npz", calibrator=f"{ART}/calibrator.npz",
                      cases_dir=f"{ART}/data/cases", cache_dir=f"{ART}/cache", report=f"{ART}/report/index.html")


# ----------------------------------------------------------------------------- local entrypoints
def _download(remote: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(vol.read_file(remote))
    local.write_bytes(data)
    return local


@app.local_entrypoint()
def upload_cases(cases_dir: str = "data/cases", remote_name: str = ""):
    """Push a cases directory to the Volume (data/cases -> /data/cases; data/cases_shortcut -> /data/cases_shortcut)."""
    remote_name = remote_name or Path(cases_dir).name
    with vol.batch_upload(force=True) as b:
        b.put_directory(cases_dir, f"/data/{remote_name}")
    print(f"uploaded {cases_dir} -> kwtc-artifacts:/data/{remote_name}")


@app.local_entrypoint()
def upload_artifacts(controller: str = "art/controller.npz", calibrator: str = "art/calibrator.npz",
                     report: str = "art/report/index.html"):
    with vol.batch_upload(force=True) as b:
        b.put_file(controller, "/controller.npz")
        if Path(calibrator).exists():
            b.put_file(calibrator, "/calibrator.npz")
        if Path(report).exists():
            b.put_file(report, "/report/index.html")
        full = Path(report).parent / "full.html"
        if full.exists():
            b.put_file(full, "/report/full.html")
    print("uploaded controller (+calibrator, +report) to kwtc-artifacts:/")


@app.local_entrypoint()
def cache(split: str = "train", model: str = "", shortcut: bool = False, k: int = 4, limit: int = 0,
          cases_dir: str = "", out_dir: str = ""):
    """Build the cache for one split with .map(), merge on the Volume, download the jsonl.

    --model gemma26b|llama8b|llama3b picks the server and the cache dir from common/models.py;
    without it the kwtc-llm secret is used with art/cache.  --shortcut uses data/cases_shortcut.
    """
    from common.io import read_jsonl

    if model:
        from common.models import get

        m = get(model)
        out_dir = out_dir or f"art/{m['cache_shortcut'] if shortcut else m['cache']}"
    cases_dir = cases_dir or ("data/cases_shortcut" if shortcut else "data/cases")
    out_dir = out_dir or ("art/cache_shortcut" if shortcut else "art/cache")
    spec = _llm_spec(model)
    cases_name, cache_name = Path(cases_dir).name, Path(out_dir).name
    ids = [r["case_id"] for r in read_jsonl(f"{cases_dir}/{split}.jsonl")]
    if limit:
        ids = ids[:limit]
    _preflight(model)
    n_ok = n_err = 0
    for res in build_case.map(ids, kwargs={"split": split, "K": k, "cases_name": cases_name, "cache_name": cache_name,
                                           "llm_spec": spec},
                              return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            n_err += 1
            print(f"  error: {str(res).strip().splitlines()[-1][:300]}")
            if n_ok == 0 and n_err >= 5:
                raise SystemExit("first 5 cases all failed; stopping (the rest would fail the same way).")
        else:
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  {n_ok}/{len(ids)} cases cached")
    remote = merge.remote(split, cache_name)
    local = _download(f"{cache_name}/{split}.jsonl", Path(out_dir) / f"{split}.jsonl")
    print(f"{split}: {n_ok} ok, {n_err} errors; merged {remote}; downloaded to {local}")


@app.local_entrypoint()
def judge(split: str = "train", model: str = "", limit: int = 0, cases_dir: str = "data/cases", out_dir: str = ""):
    """Blind-judge every committed, grounded verified verdict of a cached split (one short call each)."""
    from agent.build_judge import needs_judge
    from common.io import read_jsonl
    from data.records import StoreRegistry

    if model:
        from common.models import get

        out_dir = out_dir or f"art/{get(model)['cache']}"
    out_dir = out_dir or "art/cache"
    cache_name, cases_name = Path(out_dir).name, Path(cases_dir).name
    registry = StoreRegistry(cases_dir)
    cases = {c["case_id"]: c for c in read_jsonl(f"{cases_dir}/{split}.jsonl")}
    recs = read_jsonl(f"{out_dir}/{split}.jsonl")
    ids = [r["case_id"] for r in recs if r["case_id"] in cases and needs_judge(r, registry, cases[r["case_id"]])]
    if limit:
        ids = ids[:limit]
    print(f"{split}: {len(ids)} of {len(recs)} cached cases have a committed, grounded verified verdict to judge")
    _preflight(model)
    n_ok = n_err = 0
    for res in judge_case.map(ids, kwargs={"split": split, "cache_name": cache_name, "cases_name": cases_name, "llm_spec": _llm_spec(model)},
                              return_exceptions=True, order_outputs=False):
        if isinstance(res, Exception):
            n_err += 1
            print(f"  error: {str(res).strip().splitlines()[-1][:300]}")
            if n_ok == 0 and n_err >= 5:
                raise SystemExit("first 5 judge calls failed; stopping.")
        else:
            n_ok += 1
    remote = merge_judge.remote(split, cache_name)
    local = _download(f"{cache_name}/judge_{split}.jsonl", Path(out_dir) / f"judge_{split}.jsonl")
    print(f"{split}: {n_ok} judged, {n_err} errors; merged {remote}; downloaded to {local}")


@app.local_entrypoint()
def sweep(seeds: int = 5, out: str = "", ws: str = "0.5,1,2,4", cs: str = "0,0.025,0.05,0.1,0.2", model: str = ""):
    """100 CPU runs in parallel -> the phase diagram and the cost-accuracy frontier."""
    cache_name = "cache"
    if model:
        from common.models import get

        cache_name = get(model)["cache"]
    out = out or (f"art/models/{model}/sweep.jsonl" if model else "art/sweep.jsonl")
    grid = [(float(w), float(c), s) for w in ws.split(",") for c in cs.split(",") for s in range(seeds)]
    rows = list(train_one.starmap(grid, kwargs={"cache_name": cache_name}))
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
