# Know-When-To-Check

A frozen LLM wrapped in a 3-action controller (**answer / verify / abstain**) trained with
REINFORCE, shipped with a benchmark that catches the ways such agents cheat: shortcut
evidence, over-abstention, fabricated citations, and answers that survive the removal of
their evidence.

Pitch line: *an agent that learns when its own answer is worth checking, and a test bench
that proves it is not gaming the reward.*

Implements `Know-When-To-Check Final Pipeline Spec.pdf` (Oct 3, 2026).

## What is here

| Stage | Module | Status |
|---|---|---|
| Data: SciFact + HealthVer download, BM25 record stores, evidence-ablated twins, leak-free group split | `data/prepare.py`, `data/records.py` | done, run on the real data (train 1028 / val 286 / test-ID 488 / test-OOD 300 cases) |
| Frozen LLM behind one `chat()` interface: vLLM-on-Modal (Gemma 4 26B-A4B-it), Claude Haiku 4.5, deterministic mock | `agent/llm.py`, `agent/mock_llm.py` | done; real backends need credentials (see below) |
| Frozen prompts, hashed into the controller | `agent/prompts/` | done |
| Provisional call x2 -> six signals (+ optional `tok_prob` from vLLM log-probs) | `agent/provisional.py` | done |
| Verification: JSON tool loop, K=4, every call logged; calculator in a Modal Sandbox (or an AST-restricted local evaluator) | `agent/verify.py`, `agent/tools.py` | done |
| Counterfactual cache (all LLM calls happen here; idempotent, resumable) | `agent/build_cache.py`, `modal_app.py::cache` | done |
| Scorer: the only code that reads gold; rewards, oracle, six integrity checks | `scorer/score.py` | done |
| 3 x 6 softmax controller, REINFORCE replay, 5 seeds, median-seed shipping, `controller.npz` with provenance | `controller/policy.py`, `controller/train.py` | done |
| Separate calibrated P(correct) (logistic regression on val) with ECE / Brier | `controller/calibrate.py` | done |
| Seven policies, cluster bootstrap CIs, McNemar, headline tables, per-case logs, case cards, seven figures | `eval/evaluate.py`, `eval/policies.py`, `eval/stats.py`, `eval/figures.py` | done |
| Reward-design sweep (4 w x 5 c x 5 seeds) -> phase diagram + cost-accuracy frontier | `controller/train.py --sweep`, `modal_app.py::sweep` | done |
| One-claim inference with the evidence-removal toggle; prompt-hash / model-revision guard | `agent/infer.py` | done |
| Live demo (FastAPI + one-page UI), locally or on Modal | `demo/serve.py`, `demo/index.html`, `modal_app.py::web` | done |
| Shortcut-trained controller (stretch, policy 7) | `data/prepare.py --shortcut-variant`, `--shortcut-controller` | done |

Everything above has been run end to end on this machine with the **mock** LLM backend.
No LLM credentials or Modal token were available here, so the vLLM, Claude and Modal code
paths are written against the current SDK/API signatures and import-checked, but have not
been exercised against live services. **Numbers produced with the mock backend are
meaningless**; every cache record and `controller.npz` carries the backend/model identity so
mock results cannot be mistaken for real ones.

## Quick start (local)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m data.prepare --out data/cases --shortcut-variant   # downloads SciFact + HealthVer (~10 MB)

# pick the frozen LLM (see "Backends"); with nothing set, the mock is used and says so
export KWTC_LLM_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-...

python -m agent.build_cache --split all --workers 8          # Stage 1: the counterfactual cache
python -m controller.train --w 1 --c 0.05 --seeds 5 --out art/controller.npz
python -m controller.train --sweep --sweep-out art/sweep.jsonl
python -m eval.evaluate --split test_id  --controller art/controller.npz --figs art/figs
python -m eval.evaluate --split test_ood --controller art/controller.npz --figs art/figs_ood \
       --shortcut-controller art/controller_shortcut.npz      # optional policy 7
python -m agent.infer --claim "Aspirin reduces the risk of myocardial infarction." --controller art/controller.npz --remove-evidence
python -m demo.serve --port 8000                               # http://127.0.0.1:8000
python -m pytest -q
```

`python main.py <prepare|cache|train|sweep|evaluate|infer|demo|test> ...` forwards to the same modules.

## Modal

One-time setup (about 10 minutes):

```bash
. .venv/bin/activate
modal setup                                       # browser login, writes ~/.modal.toml
# accept the Gemma licence on huggingface.co, create a read token, then:
modal secret create huggingface-secret HF_TOKEN=hf_...
modal deploy agent/serve_vllm.py                  # 1. frozen LLM: vllm serve on an H200, revision-pinned, --max-logprobs 20
modal run agent/serve_vllm.py                     #    waits for /health, sends one JSON completion with log-probs, prints the URL
modal secret create kwtc-llm KWTC_LLM_BACKEND=vllm KWTC_VLLM_URL=https://<workspace>--kwtc-vllm-server.modal.run
#   fallback: modal secret create kwtc-llm KWTC_LLM_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-...
```

Then either run everything with one script

```bash
scripts/run_modal.sh gate      # upload data, 20 cases end to end, health report (Gate 1), stop
scripts/run_modal.sh all       # caches (incl. shortcut variant), training, sweep, evaluation, figures, demo deploy
```

or step by step:

```bash
modal run modal_app.py::upload_cases --cases-dir data/cases            # data -> Volume kwtc-artifacts:/data/cases
modal run modal_app.py::upload_cases --cases-dir data/cases_shortcut
modal run modal_app.py::cache --split train --limit 20                 # Gate 1
python -m agent.cache_report --split train                             # JSON validity, signals, tool use (no gold)
modal run modal_app.py::cache --split train                            # 2. build_case.map(); then val, test_id, test_ood
modal run modal_app.py::cache --split train --cases-dir data/cases_shortcut --out-dir art/cache_shortcut   # (+ val) policy 7
python -m controller.train --w 1 --c 0.05 --seeds 5 --out art/controller.npz
modal run modal_app.py::calc --expression "2**10"                      # 3. sandboxed calculator smoke test
modal run modal_app.py::sweep                                          # 4. train_one.starmap() over 4 w x 5 c x 5 seeds -> art/sweep.jsonl
python -m eval.evaluate --split test_id  --controller art/controller.npz --figs art/figs
python -m eval.evaluate --split test_ood --controller art/controller.npz --figs art/figs_ood --shortcut-controller art/controller_shortcut.npz
modal run modal_app.py::upload_artifacts                               # controller.npz + calibrator.npz -> Volume
modal deploy modal_app.py                                              # 5. live demo URL (FastAPI via @modal.asgi_app)
```

Cache records are written per case to the Volume (`/art/cache/<split>/<case_id>.json`), so
`cache` is idempotent and resumable; `merge` produces `<split>.jsonl`, which the entrypoint
downloads to `art/cache/`.  Each `cache` invocation also carries the vLLM-side `tok_prob`
log-prob feature automatically when the backend is vLLM.

## Epistemic discipline (why the agent's "checking" was broken, and the fix)

Measured on the real caches, verification *degraded* belief: when the verifier changed its mind
the new verdict was right 20-33 % of the time while the discarded prior was right 58-63 %. Almost
all of the damage was "insufficient evidence" turned into a confident verdict after reading a
loosely related abstract, and for the 3B model half of the verified verdicts cited nothing the agent
had actually read. A 3-action controller trained on such a check correctly learns never to check -
which is a sound decision, but the epistemic claim of the project ("learns when its answer is worth
checking") then has nothing to stand on. The fix is three rules plus one new component, all in
`agent/epistemic.py`, `agent/judge.py`, `controller/epistemic.py`:

1. **No evidence, no verdict.** A supported/refuted verdict must cite evidence the agent actually
   saw: a shown abstract for the provisional answer, a sentence of an opened record for the checked
   answer. Otherwise the justified output is `insufficient_evidence`. Deterministic and auditable.
   For single-sentence records (HealthVer, Climate-FEVER, VitaminC snippets) any citation of the
   record counts as citing its sentence, whatever index the model wrote; Gemma numbers from 1, and
   without this rule 78-84 % of its grounded checks on those sets were wrongly rejected. After
   changing it, re-run `modal run modal_app.py::judge --model gemma26b --split <split>` so the newly
   eligible cases get a judge record (the pass is idempotent and only judges the new ones).
2. **Judgement is separated from search.** The *blind judge* (`modal run modal_app.py::judge --model
   <m>`; one short call per committed verified verdict) sees only the claim and the cited sentences,
   with no search narrative, and its reading replaces the verifier's self-assessment. Its top-k token
   probabilities over the three relation labels give a credence that is a probability distribution.
3. **A check is evidence, not an oracle.** After checking, a second decision uses what the check
   revealed (does the new verdict agree with the prior, was it grounded, how confident, what the
   judge said) and commits the checked verdict, keeps the prior, or abstains (paying the tool cost).
   Agreement between prior and check is the strongest reliability signal in the data.
4. **Decisions follow from calibrated credences.** `EUController` fits P(answer right), P(checked
   verdict right) and P(prior right | check) on validation and takes the action with the highest
   expected reward under (w, c): "answer iff P(correct) > w/(1+w)" is the literal mechanism and the
   number the agent reports is its credence. `TwoStageController` is the REINFORCE counterpart with
   two softmax heads (3 x 6 pre-check, 3 x 13 post-check), trained by the same cache replay.

Evaluation adds the policies *Always check (disciplined verifier)*, *Epistemic agent (two-stage
RL)* and *Epistemic agent (credence-based)*, a second phase diagram for the credence-based agent,
check / contested rates, and the ECE of the reported credence. `scripts/run_models.sh <model> judge`
runs the blind-judge pass and re-evaluates; `art/models/comparison.md` has the cross-model table.

## Scaling experiment: Gemma 4 26B-A4B vs Llama 3.1 8B vs Llama 3.2 3B

Question: does deciding when to check pay off more for less capable frozen models?  Every
model runs the same cases, prompts, tools and K; only the frozen LLM changes.  Models,
pinned revisions, GPUs and cache names live in `common/models.py`; each model has its own
server class in `agent/serve_vllm.py` (app `kwtc-vllm`), its own cache (`art/cache`,
`art/cache_llama8b`, `art/cache_llama3b`) and its own results in `art/models/<model>/`.

```bash
# once: the Hugging Face account in `huggingface-secret` must have access to both meta-llama repos
scripts/run_models.sh deploy            # registers all three servers; GPUs start only on demand
scripts/run_models.sh llama8b gate      # preflight + 20 cases + health report
scripts/run_models.sh llama8b all       # caches, controllers, sweep, evaluation
scripts/run_models.sh llama3b all
scripts/run_models.sh compare           # art/models/comparison.md (+ _ood) and art/models/figs_compare/
```

Gemma's results are already in `art/models/gemma26b/` (copied from the first run, not rebuilt).
Gemma 4 26B-A4B is a mixture of experts with about 4B parameters active per token, so the
comparison lists total and active size.  Three models from two families cannot separate size
from family or training recipe, so the size trend is descriptive.

## Cross-domain epistemics, the evidence ladder, and the selection ablation

In-domain, the epistemic agent works; out of domain (HealthVer) its calibration collapsed. Three
additions turn that from a dead end into measurements, following the literature on calibration
under shift (Ovadia et al. 2019; Li et al. 2024 "Few-Shot Recalibration of Language Models"),
conformal risk control for selective prediction (Angelopoulos & Bates), consistency-based
uncertainty (Farquhar et al. 2024, semantic entropy; Kadavath et al. 2022, P(True)) and contrastive
evidence (Schuster et al. 2021, VitaminC; Saakyan et al. 2021, COVID-Fact).

**More domains** (`data/extra.py`, no change to the existing splits):

| Split | Source | Why it is a fair out-of-domain test |
|---|---|---|
| `test_climate` | Climate-FEVER (1,535 real climate claims, 5 labelled Wikipedia sentences each) | the agent sees 3 of the 5 sentences, so checking can find the decisive one; "not enough info" claims carry evidence that is present but inconclusive; DISPUTED claims excluded |
| `test_vitc` | VitaminC test (contrastive Wikipedia revision pairs) | small factual changes flip the label; records = the other sentences of the same page |
| `test_ladder` | SciFact dev, rationale-removed variants | see the ladder below |

```bash
python -m data.extra --out data/cases                 # builds only the new splits (300 / 300 / 188 cases)
scripts/run_models.sh llama8b extra                   # caches + blind judge for the new splits, then the two reports below
python -m eval.transfer                               # art/models/transfer.md  + art/models/figs_transfer/
python -m eval.ladder                                 # art/models/ladder.md    + art/models/figs_ladder/
python -m eval.ablation_selection                     # art/models/ablation_selection.md
```

**`eval/transfer.py`: does the agent know what it does not know in a new domain?** For every
model and out-of-domain split: the in-domain credence-based agent zero-shot; *few-shot
recalibration* (a two-parameter Platt rescaling of its credences fit on k = 25 / 50 / 100 labelled
cases of the new domain, held-out evaluation, 20 random draws) with an in-sample ceiling;
*risk control* (on the same k cases, the largest coverage whose 90 % Hoeffding bound on the error
rate of delivered verdicts is below a target alpha, then the realised error on held-out cases);
and a *novelty detector* (Mahalanobis distance of the six pre-check signals to the SciFact training
distribution: AUROC for new-domain vs in-domain cases, calibration error by novelty quartile, and
the effect of abstaining on cases flagged as novel). If calibration error drops with a few dozen
labelled cases, the signals transfer and only their scale was wrong.

**`eval/ladder.py`: does confidence track evidence or topic?** Each supported/refuted SciFact
test claim appears at three evidence levels: L0 full evidence, L1 the gold abstract shown with its
rationale sentences deleted (topic intact, decisive information gone; label insufficient), L2 the
abstract removed (the twin). Reports the share of cases where each policy commits to a verdict at
each level, the credence it attaches, per-claim monotonicity, and *contrastive pairs*: SciFact
claims citing the same abstract with opposite labels (23 in test, 185 in train), where reading the
evidence forces different verdicts. Information-theoretic reading: label information is high at L0
and zero at L1 and L2 while lexical overlap barely changes between L0 and L1.

**`eval/ablation_selection.py`** tests idea 2: training the controllers only on decision-relevant
cases (first answer and disciplined check disagree in correctness) or up-weighting them.

## Robustness notes (learned from the Modal runs)

* **Bounded tool outputs.** `read_record` returns at most 25 sentences (indices preserved, the rest
  noted) and sentences are capped at 400 characters; the verify loop re-sends the whole conversation
  every turn, and a small model reading several 40-60-sentence structured abstracts pushed one prompt
  past Llama's 16k context. Only 0.2-0.4 % of previously cached cases read such a record, so the
  existing caches remain comparable.
* **Context guard.** When the conversation exceeds ~36k characters the loop forces the final answer;
  if the backend still rejects the prompt as too long, the provisional verdict stands and the record is
  flagged `context_overflow` instead of failing the case.
* **Calculator sandbox.** `modal.Sandbox.create(timeout=...)` is the sandbox's lifetime, not the
  command timeout; it is now 60 s with a 10 s exec limit (a 10 s lifetime expired before `exec` ran).
* **Long runs.** The scripts use `caffeinate -dims` and `modal run --detach` for cache and judge maps, so
  a sleeping or disconnected laptop no longer kills a run; finished cases are on the Volume either
  way and a re-run only merges and downloads them. Modal preemptions ("Container terminated due to
  preemption") are retried by Modal and are harmless here.
* **Gemma cites sentences from 1.** See the grounding rule note above.

## Backends

| `KWTC_LLM_BACKEND` | What it uses | Notes |
|---|---|---|
| `vllm` | OpenAI-compatible vLLM server at `KWTC_VLLM_URL`, served name `KWTC_VLLM_MODEL` (default `llm`) | guided JSON decoding, `seed`, thinking off, per-token log-probs -> `tok_prob` feature (`--tok-prob` at cache build, `--use-tok-prob` at train) |
| `claude` | Anthropic SDK, `KWTC_CLAUDE_MODEL` (default `claude-haiku-4-5`) | structured outputs for the provisional JSON; no seed parameter (the two samples differ by sampling); no log-probs |
| `mock` | deterministic lexical-overlap stand-in | for tests and for building the trainer/scorer/figures before the real cache exists |

Auto-detection when unset: vLLM if `KWTC_VLLM_URL` is set, Claude if an Anthropic credential
is set, otherwise mock (with a warning on stderr).

The calculator tool runs in a Modal Sandbox (`block_network=True`, 10 s timeout) when
`KWTC_CALC_BACKEND=modal` (set inside the Modal image). Locally it runs an AST-whitelisted
arithmetic evaluator (`agent/tools.py::safe_calculate`): numbers, operators, `math`
functions, nothing else.

## Layout

```
data/prepare.py        download SciFact + HealthVer, BM25 record files, twins, group split  (writes data/cases/)
data/records.py        RecordStore: BM25 with per-case exclusions (ablated twins), StoreRegistry
agent/llm.py           one chat() interface: vLLM-on-Modal | Claude API | mock
agent/prompts/         provisional.txt, verify.txt (hashed, frozen after the cache is built)
agent/provisional.py   two samples -> provisional answer + six signals
agent/verify.py        JSON tool loop, K limit, forced final, malformed-JSON retry
agent/tools.py         search_records / read_record / calculate with a full call log
agent/build_cache.py   Stage 1 cache builder (local, threaded, resumable)
agent/infer.py         deployment path for one claim (+ evidence-removal twin run)
agent/serve_vllm.py    Modal @app.server running vllm serve (Gemma 4 26B-A4B-it, pinned revision)
controller/policy.py   softmax(W x), z-scoring, controller.npz save/load with hash guards
controller/train.py    REINFORCE replay, model selection on val, median-seed shipping, sweep
controller/calibrate.py logistic P(correct), ECE, Brier, reliability bins
scorer/score.py        gold labels live only here: correctness, rewards, oracle, integrity checks
eval/evaluate.py       policies, metrics, bootstrap, McNemar, tables, logs, cards, figures
eval/figures.py        the seven figures
modal_app.py           cache map, merge, sandboxed calc, sweep starmap, demo endpoint
demo/serve.py, demo/index.html   live demo
tests/                 unit tests + an end-to-end mock run on a synthetic corpus
art/                   outputs: cache/, controller.npz, calibrator.npz, sweep.jsonl, results_*.md/json, logs/, figs/
```

## Data

* SciFact: 5,183 abstracts; official train (809 claims: 332 S / 173 C / 304 NEI) and dev
  (300: 124 / 64 / 112). Claims citing the same abstract are joined by union-find; whole
  groups go to one split (80 % of groups -> train, 20 % -> val). Dev is test-ID.
* Every SUPPORT/CONTRADICT claim gets an evidence-ablated twin: same claim, gold abstracts
  removed from that case's accessible records, label `insufficient_evidence`. Twins stay in
  their parent's split and group.
* NEI cases get the same top-3 BM25 retrieval as everyone else, so they never carry an empty
  evidence field (the SciFact shortcut).
* HealthVer test is the out-of-domain split: initial evidence = the pair's snippet,
  accessible records = the other snippets under the same question, grouped by question.
  **Deviation from the spec:** HealthVer test has only 230 unique claims, so 300 cases with
  one pair per claim is impossible. `prepare.py` takes one pair per claim first (stratified
  by label, 100 / 100 / 100) and fills the rest with a second, different-evidence pair for 70
  claims. Set `--ood-n 230` for strictly one pair per claim.
* `data/cases/gold/` holds labels, gold doc ids and gold rationale sentences; only
  `scorer/score.py` reads it. Agent inputs never contain gold.
* Known caveat from the spec, still open: an ablated twin may be supported by a non-cited
  abstract. Spot-check 30 twins by hand (`data/cases/*.jsonl`, variant `ablated`) and report
  the error rate.
* `--shortcut-variant` also writes `data/cases_shortcut/` with raw SciFact semantics (NEI
  cases see no evidence, S/R cases see their gold abstract, no twins) for policy 7.

## Signals, controller, reward

Features: `bias, conf_mean, agree, suff_mean, conf_gap, says_insuff` (+ `tok_prob` with
vLLM). Non-bias features are z-scored with train statistics stored next to W.
Policy: `pi(a|x) = softmax(W x)`, W is 3 x 6, zeros at init; sampling in training, argmax at
deployment. Reward: answer +1 / -w, verify +1 / -w minus c per tool call, abstain 0. Answer
beats abstain when P(correct) > w/(1+w). Defaults w = 1, c = 0.05, K = 4.

`controller.npz` stores W, feature mean/std, feature and action names, w, c, K, the prompt
hash, the LLM model id and revision, the cache hash, the seeds, and the shipped (median)
seed. `Controller.load` refuses a changed prompt; `agent.infer` refuses a different model or
revision (`KWTC_SKIP_HASH_CHECK=1` or `--allow-mismatch` overrides).

## Evaluation outputs

`python -m eval.evaluate --split <split> --figs <dir>` writes

* `art/results_<split>.md` / `.json`: a self-contained report. It opens with a plain-English
  "What this report says" summary generated from the numbers, then "How to read this report"
  (setup, the three actions, how utility is scored, what the brackets mean, one line per
  policy), then four captioned tables that share one vocabulary (`eval/report.py`): Table 1
  how good each policy is, Table 2 integrity (cheating, luck, memory), Table 3 cost and
  calibration, Table 4 what the epistemic agents do; each table ends with a glossary of its
  columns. Then statistics (paired differences with a verdict on whether they are real,
  McNemar, seed spread), a figure guide, and three case cards.
* `art/logs/<split>_<policy>.jsonl`: one line per case in the spec's log format; `gold`,
  `correct`, `reward`, `integrity` are added by the scorer and never exist at deployment.
* `art/calibrator.npz`: the P(correct) model fit on val.
* Figures: `1_risk_coverage`, `2_phase_diagram`, `2b_phase_diagram_epistemic`,
  `3_grounding_test` (splits with twins), `4_reliability`, `5_cost_accuracy_frontier`,
  `6_W_heatmap`, `7_action_mix`. Every figure carries a "How to read" caption in the image.
* `art/models/comparison.md` (+ `_ood`): the cross-model report with the same structure
  (summary, how to read, six captioned tables, size-trend reading) and
  `art/models/figs_compare/`.

Integrity definitions (per policy, from logs + gold): *fabrication* = a cited doc id that was
never shown or opened; *right for the wrong reason* = correct S/R verdict with no cited
sentence in the gold rationale (doc-level for the answer action, which cites doc ids only);
*grounding flip* = on parent/twin pairs with a correct parent, the twin ends
insufficient_evidence or abstains; *stubborn* = the twin keeps the parent's S/R verdict with
confidence >= 0.7; *verification flips* = wrong->right and right->wrong counts after verify;
*unnecessary abstention* = abstained although answer or verify would have been right.

## Fallbacks (from the spec)

| Trigger | What to do |
|---|---|
| vLLM not serving | `KWTC_LLM_BACKEND=claude`; drop `--tok-prob` / `--use-tok-prob` |
| Cache too slow | `--k 2`, `--limit`, `prepare.py --ood-n 150 --no-twins` (twins on test only by hand-editing splits) |
| Gemma JSON keeps breaking | guided decoding is already on for the provisional call; lower `PROVISIONAL_TEMPERATURE` in `common/config.py` |
| Controller collapses to one action | check the reward scale, raise `--ent`; the phase diagram shows the collapse as a finding |
| Demo endpoint fails | `GET /api/cached/<case_id>` replays cache records; the static page still renders them |
| Shortcut controller not done | skip `--shortcut-controller`; mention the shortcut and the HealthVer drop in one line |
