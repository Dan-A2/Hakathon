"""FastAPI app for the live demo (served by Modal via modal_app.web, or locally):

    python -m demo.serve --port 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from common import config as C
from common.io import read_jsonl

INDEX = Path(__file__).parent / "index.html"


class InferRequest(BaseModel):
    claim: str | None = None
    case_id: str | None = None
    remove_evidence: bool = False


def curated_claims(cases_dir: Path | str, split: str = "test_id", n: int = 10) -> tuple[list[dict], dict[str, dict]]:
    """Ten test claims that have an evidence-ablated twin (so the toggle has something to remove)."""
    cases = {r["case_id"]: r for r in read_jsonl(Path(cases_dir) / f"{split}.jsonl")}
    out = []
    for cid, c in sorted(cases.items()):
        if c.get("variant") == "base" and (cid + "-abl") in cases:
            out.append({"case_id": cid, "claim": c["question"], "twin_id": cid + "-abl"})
        if len(out) >= n:
            break
    return out, cases


def create_app(controller: str | Path = C.ART_DIR / "controller.npz", calibrator: str | Path | None = None,
               cases_dir: str | Path = C.CASES_DIR, backend: str | None = None, cache_dir: str | Path | None = C.CACHE_DIR,
               split: str = "test_id") -> FastAPI:
    app = FastAPI(title="Know-When-To-Check demo")
    state: dict = {"rt": None, "error": None}
    curated, cases = curated_claims(cases_dir, split)
    cached: dict[str, dict] = {}
    if cache_dir and (Path(cache_dir) / f"{split}.jsonl").exists():
        cached = {r["case_id"]: r for r in read_jsonl(Path(cache_dir) / f"{split}.jsonl")}

    def runtime():
        if state["rt"] is None:
            from agent.infer import Runtime

            try:
                state["rt"] = Runtime(controller, calibrator, backend, cases_dir)
            except Exception as e:  # noqa: BLE001 - surface the reason to the UI
                state["error"] = repr(e)
                raise HTTPException(503, f"agent not ready: {e!r}") from e
        return state["rt"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        try:
            rt = runtime()
            return {"ok": True, "llm": rt.llm.identity(), "controller": {"w": rt.ctrl.w, "c": rt.ctrl.c, "K": rt.ctrl.K,
                    "prompt_hash": rt.ctrl.prompt_hash}, "calibrator": rt.cal is not None, "curated": len(curated),
                    "cached_cases": len(cached)}
        except HTTPException as e:
            return {"ok": False, "error": e.detail}

    @app.get("/api/claims")
    def claims():
        return curated

    @app.post("/api/infer")
    def infer(req: InferRequest):
        from agent.infer import infer_case, infer_claim

        rt = runtime()
        if req.case_id:
            if req.case_id not in cases:
                raise HTTPException(404, f"unknown case {req.case_id}")
            target = req.case_id + "-abl" if req.remove_evidence and (req.case_id + "-abl") in cases else req.case_id
            res = infer_case(rt, cases[target])
            return {"mode": "case", "result": res, "removed_doc_ids": res["excluded_doc_ids"]}
        if not req.claim or not req.claim.strip():
            raise HTTPException(400, "claim or case_id is required")
        res = infer_claim(rt, req.claim.strip(), remove_evidence=req.remove_evidence)
        if req.remove_evidence:
            return {"mode": "live", "result": res["evidence_removed"], "removed_doc_ids": res["removed_doc_ids"],
                    "with_evidence": res["with_evidence"]}
        return {"mode": "live", "result": res, "removed_doc_ids": []}

    @app.get("/api/cached/{case_id}")
    def cached_case(case_id: str):
        """Replay of the cache record (fallback when the live LLM is unavailable)."""
        if case_id not in cached:
            raise HTTPException(404, "not in cache")
        return cached[case_id]

    return app


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--controller", default=str(C.ART_DIR / "controller.npz"))
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--backend", default=None)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(args.controller, args.calibrator, args.cases_dir, args.backend, args.cache_dir),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
