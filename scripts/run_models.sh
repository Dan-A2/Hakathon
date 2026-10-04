#!/usr/bin/env bash
# Scaling experiment: run the full pipeline for one frozen model, then compare models.
#
#   scripts/run_models.sh deploy                 # register all vLLM servers (GPUs start only on demand)
#   scripts/run_models.sh llama8b gate           # preflight + 20 cases + health report
#   scripts/run_models.sh llama8b all            # caches, blind judge, training, sweep, evaluation for that model
#   scripts/run_models.sh llama8b judge          # only the blind-judge pass + retrain + re-evaluate (cache exists)
#   KWTC_JUDGE=0 scripts/run_models.sh llama8b all   # skip the judge
#   scripts/run_models.sh compare                # cross-model tables and figures (art/models/)
#   scripts/run_models.sh llama8b extra          # Climate-FEVER + VitaminC + evidence ladder caches, then transfer/ladder reports
#
# Models: gemma26b, llama8b, llama3b (common/models.py).  Llama weights are gated: request access on
# huggingface.co for the account whose token is in the `huggingface-secret` Modal secret.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=${PY:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}
step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

case "${1:-}" in
  deploy)  $MODAL deploy agent/serve_vllm.py; exit 0 ;;
  compare) $PY -m eval.compare --split test_id; $PY -m eval.compare --split test_ood; exit 0 ;;
  gemma26b|llama8b|llama3b) MODEL=$1 ;;
  *) echo "usage: $0 deploy | compare | <gemma26b|llama8b|llama3b> [gate|all]"; exit 2 ;;
esac
MODE=${2:-all}
if [ "$MODE" = "extra" ]; then
  # more domains + the evidence ladder: caches, blind judge, then the cross-domain and ladder reports
  CACHE=art/$($PY -c "from common.models import get; print(get('$MODEL')['cache'])")
  [ -f data/cases/test_climate.jsonl ] || $PY -m data.extra --out data/cases
  $MODAL run modal_app.py::upload_cases --cases-dir data/cases
  for s in test_climate test_vitc test_ladder; do $MODAL run --detach modal_app.py::cache --model "$MODEL" --split "$s"; done
  if [ "${KWTC_JUDGE:-1}" = "1" ]; then for s in test_climate test_vitc test_ladder; do $MODAL run --detach modal_app.py::judge --model "$MODEL" --split "$s"; done; fi
  OUT=art/models/$MODEL
  for s in test_climate test_vitc; do
    $PY -m agent.cache_report --split "$s" --cache-dir "$CACHE"
    $PY -m eval.evaluate --split "$s" --controller "$OUT/controller.npz" --cache-dir "$CACHE" --out "$OUT" --logs "$OUT/logs" --figs "$OUT/figs_$s" --sweep "$OUT/sweep.jsonl"
  done
  $PY -m eval.transfer && $PY -m eval.ladder
  for s in test_climate test_vitc; do $PY -m eval.compare --split "$s" || true; done
  exit 0
fi
if [ "$MODE" = "judge" ]; then
  CACHE=art/$($PY -c "from common.models import get; print(get('$MODEL')['cache'])")
  OUT=art/models/$MODEL
  for s in train val test_id test_ood; do $MODAL run --detach modal_app.py::judge --model "$MODEL" --split "$s"; done
  $PY -m controller.epistemic --w 1 --c 0.05 --seeds 5 --cache-dir "$CACHE" --out-dir "$OUT"
  $PY -m eval.evaluate --split test_id  --controller "$OUT/controller.npz" --cache-dir "$CACHE" --out "$OUT" --logs "$OUT/logs" --figs "$OUT/figs" --sweep "$OUT/sweep.jsonl"
  $PY -m eval.evaluate --split test_ood --controller "$OUT/controller.npz" --cache-dir "$CACHE" --out "$OUT" --logs "$OUT/logs" --figs "$OUT/figs_ood" --sweep "$OUT/sweep.jsonl" --shortcut-controller "$OUT/controller_shortcut.npz"
  $PY -m eval.compare --split test_id; exit 0
fi
CACHE=art/$($PY -c "from common.models import get; print(get('$MODEL')['cache'])")
CACHE_SC=art/$($PY -c "from common.models import get; print(get('$MODEL')['cache_shortcut'])")
OUT=art/models/$MODEL
mkdir -p "$OUT"

# Keep the Mac awake for the whole run: a sleeping laptop disconnects the client and Modal stops the app.
if [ -z "${KWTC_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export KWTC_CAFFEINATED=1; exec caffeinate -dims "$0" "$@"     # no display/idle/disk/system sleep while this runs
fi

step "$MODEL: preflight against the deployed server (health, one real call)"
$MODAL run modal_app.py::preflight --model "$MODEL"

step "$MODEL: Gate 1, 20 train cases"
$MODAL run --detach modal_app.py::cache --model "$MODEL" --split train --limit 20
$PY -m agent.cache_report --split train --cache-dir "$CACHE"
if [ "$MODE" = "gate" ]; then echo; echo "Gate done. If it looks healthy: $0 $MODEL all"; exit 0; fi

step "$MODEL: full counterfactual cache"
for s in train val test_id test_ood; do $MODAL run --detach modal_app.py::cache --model "$MODEL" --split "$s"; done
for s in train val; do $MODAL run --detach modal_app.py::cache --model "$MODEL" --shortcut --split "$s"; done
$PY -m agent.cache_report --split test_id --cache-dir "$CACHE"

if [ "${KWTC_JUDGE:-1}" = "1" ]; then
  step "$MODEL: blind judge (evidence-only reading of every committed verified verdict)"
  for s in train val test_id test_ood; do $MODAL run --detach modal_app.py::judge --model "$MODEL" --split "$s"; done
fi

step "$MODEL: train controllers (3-action REINFORCE, two-stage epistemic RL, credence-based)"
$PY -m controller.train --w 1 --c 0.05 --seeds 5 --cache-dir "$CACHE" --out "$OUT/controller.npz"
$PY -m controller.train --w 1 --c 0.05 --seeds 5 --cases-dir data/cases_shortcut --cache-dir "$CACHE_SC" --out "$OUT/controller_shortcut.npz"
$PY -m controller.epistemic --w 1 --c 0.05 --seeds 5 --cache-dir "$CACHE" --out-dir "$OUT"

step "$MODEL: reward-design sweep on Modal"
$MODAL run modal_app.py::sweep --model "$MODEL" --seeds 5 --out "$OUT/sweep.jsonl"

step "$MODEL: evaluate"
$PY -m eval.evaluate --split test_id  --controller "$OUT/controller.npz" --cache-dir "$CACHE" --out "$OUT" --logs "$OUT/logs" --figs "$OUT/figs" --sweep "$OUT/sweep.jsonl"
$PY -m eval.evaluate --split test_ood --controller "$OUT/controller.npz" --cache-dir "$CACHE" --out "$OUT" --logs "$OUT/logs" --figs "$OUT/figs_ood" --sweep "$OUT/sweep.jsonl" --shortcut-controller "$OUT/controller_shortcut.npz"

step "cross-model comparison so far"
$PY -m eval.compare --split test_id
echo; echo "Done: $OUT/results_test_id.md, $OUT/results_test_ood.md, art/models/comparison.md"
