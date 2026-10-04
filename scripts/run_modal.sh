#!/usr/bin/env bash
# Reproduce every result on Modal.  Prerequisites (one-time, see README "Modal"):
#   modal setup
#   modal secret create huggingface-secret HF_TOKEN=hf_...          # Gemma licence accepted on huggingface.co
#   modal deploy agent/serve_vllm.py && modal run agent/serve_vllm.py
#   modal secret create kwtc-llm KWTC_LLM_BACKEND=vllm KWTC_VLLM_URL=https://<workspace>--kwtc-vllm-server.modal.run
#
# Usage:  scripts/run_modal.sh gate     # upload data, run 20 cases end to end, print the health report, stop
#         scripts/run_modal.sh all      # everything: caches, training, sweep, evaluation, figures, demo deploy
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
MODE=${1:-all}

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

[ -d data/cases ] || { step "prepare data"; $PY -m data.prepare --out data/cases --shortcut-variant; }

step "upload cases to the Volume"
$MODAL run modal_app.py::upload_cases --cases-dir data/cases
$MODAL run modal_app.py::upload_cases --cases-dir data/cases_shortcut

step "Gate 1: 20 train cases end to end"
$MODAL run modal_app.py::cache --split train --limit 20
$PY -m agent.cache_report --split train
if [ "$MODE" = "gate" ]; then echo; echo "Gate check done. Re-run with 'all' to build everything."; exit 0; fi

step "full counterfactual cache (critical path)"
for s in train val test_id test_ood; do $MODAL run modal_app.py::cache --split "$s"; done
for s in train val; do $MODAL run modal_app.py::cache --split "$s" --cases-dir data/cases_shortcut --out-dir art/cache_shortcut; done
$PY -m agent.cache_report --split train

step "train the controller (local NumPy, seconds)"
$PY -m controller.train --w 1 --c 0.05 --seeds 5 --out art/controller.npz
$PY -m controller.train --w 1 --c 0.05 --seeds 5 --use-tok-prob --out art/controller_tokprob.npz || echo "(tok_prob ablation skipped: no log-probs in cache)"
$PY -m controller.train --w 1 --c 0.05 --seeds 5 --cases-dir data/cases_shortcut --cache-dir art/cache_shortcut --out art/controller_shortcut.npz

step "reward-design sweep on Modal (100 CPU runs)"
$MODAL run modal_app.py::sweep --seeds 5 --out art/sweep.jsonl

step "evaluate (local): tables, logs, cards, figures"
$PY -m eval.evaluate --split test_id  --controller art/controller.npz --figs art/figs
$PY -m eval.evaluate --split test_ood --controller art/controller.npz --figs art/figs_ood --shortcut-controller art/controller_shortcut.npz

step "deploy the live demo"
$MODAL run modal_app.py::upload_artifacts
$MODAL deploy modal_app.py

echo
echo "Done. Results: art/results_test_id.md, art/results_test_ood.md, art/figs/, art/figs_ood/, art/logs/, art/cards_*.md"
