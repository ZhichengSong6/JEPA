# CEM-MH follow-up diagnostic

## Scope

No training, checkpoint, original evaluation script, cost function, or CEM update
is changed. The driver instantiates the installed official `CEMSolver`. A proxy
records its calls/return values and delegates computation unchanged. It targets
the recorded server version `stable-worldmodel==0.0.6` and fails on a version
change rather than silently comparing different planners.

For each seed 42–46, retain all 100 original dataset starts and B1000
(N=100, I=10, K=10). Read the previous `cem/*.json` files, including actual policy
paths. Select all historical discordant cases plus two common-failure and two
common-success controls per seed, using a fixed independent selection RNG.
Do not change the world to contain only selected cases: that changes CEM RNG
allocation. Old labels remain recorded even if the new cohort reclassifies them.

## A new matched-reset cohort, not recovered historical variations

The old run did not record live variations. They cannot be recovered after the
fact. This diagnostic sets an explicit native reset seed (`seed*1000+env_index`)
ONLY when the dataset does not supply one. Other reset arguments and dataset
state/goal calls pass through unchanged. It captures native reset arguments,
external `_set_state`/`_set_goal_state` calls, all live variations, initial render
hashes, and full observed/body physics (including block velocity/angular velocity).
Each subsequent run must match the initial variations, images, and physics.

Four runs per seed:

1. MH repeat 0, passive cost/plan tracing on selected cases.
2. MH repeat 1, official CEM without the cost/solver proxy.
3. CEM-MH repeat 0, passive cost/plan tracing on selected cases.
4. CEM-MH repeat 1, official CEM without the cost/solver proxy.

Both repeats still record native actions and states. A changed success, action,
physics trajectory, reset count, or initial context blocks mechanism analysis.
The files and error report remain available. This is a repeatability gate, not a
claim of universal simulator determinism or a new headline success benchmark.

## Counterfactual replay and aligned endpoints

Never reconstruct a replan just by calling `_set_state(solve_pose)`: that can lose
unobserved dynamics. Re-run the recorded native reset and external setter recipe,
then the full actually executed native action prefix. Check physics at EVERY
prefix step and the rendered solve-start image. At the first solve, validate
policy pixels against the original DATASET start frame, which World injects
after reset. At subsequent solves, validate against AddPixelsWrapper output.
Do not require a dataset start image to equal the live render. Also replay the actually returned action plan
and require agreement with recorded live native actions and states.

If any check fails, STOP. No approximate oracle is silently substituted.

After these checks, replay candidate actions for exactly 25 native steps. Call
native PushT directly, without a TimeLimit or auto-reset wrapper. Continue after
`terminated=True` only for this fixed-endpoint diagnostic, not in closed-loop
evaluation. Any native truncation is an error. Save first success step,
ever-success, fixed-endpoint success, endpoint state, and the installed
`raw.eval_state` endpoint distance. Encoder cost always uses the same fixed
endpoint and exact processed dataset goal; it is not compared to an early-stop
render. Official endpoint distance is recorded separately from latent cost.

## Cross-scoring and returned means

Use both models' traced histories/populations. On EACH source history and fixed
population, score BOTH models. Do not combine candidates from different states
and call them a common population. Verify source-model rescoring and device
`torch.topk` elite indices against the live trace. Assert identical frozen
encoder/projector parameters before using a shared encoder oracle.

Record candidate coverage, selected success, elite success retention, overlap
with exact-encoder elite, cost gaps at the elite boundary, and selection regret.
Physically replay three elite means: MH-scored, CEM-MH-scored, encoder-scored,
all computed from that same population using device float32 mean arithmetic.
These are offline interventions, not additional closed-loop success claims.
At the final iteration, source elite mean is the official returned plan; the
live-action agreement check independently verifies the actual return path.

This stage does NOT yet run paired-noise next-generation CEM populations or a
uniform-weight training ablation. It supplies the evidence to choose those
experiments rather than changing training before checking repeatability.

## Commands (run in the repository; Bash submission stays on the server)

```bash
python -m unittest discover -s tests -p 'test_cem_mh_diagnostics.py' -v
python eval_cem_mh_diagnostics.py \
  --input-dir /absolute/path/to/previous_run/cem \
  --output-dir /absolute/path/to/new_run/seed42 --seed 42 --stage all
python scripts/summarize_cem_mh_diagnostics.py --root /absolute/path/to/new_run
```

`--stage capture` runs only repeatability/tracing. `--stage analyze` analyzes an
existing passed capture with matching checkpoint hashes. Analysis refuses an
existing `populations/` directory: preserve failed artifacts and start a new
analysis output only after investigating the cause; do not quietly mix reruns.
Default iterations are zero-based 0, 5, 9. `--controls` changes control count but
never drops historical discordants. Do not treat selected-population averages
as unbiased benchmark estimates.

Outputs per seed: protocol/manifest JSON, four full run records, repeatability
report, two local `.pt` traces, per-population JSON/NPZ, mechanism JSON, or a
`BLOCKED.json`. Summary produces case and population CSVs and `results.tar.gz`.
Large `.pt` traces stay on the server. Only load `.pt` files this driver generated
locally; they use Python serialization and are not safe for untrusted inputs.

## Validation status

CPU mock tests cover reset recipe/prefix replay, hidden state checks, temporal
alignment, truncation rejection, manifest selection, repeatability blocking,
fresh cross-scoring inputs, auto-reset segment retention, and a 100-env official-
interface pass-through fixture. They are not real GPU/PushT execution. Runtime
checks deliberately fail closed when the installed environment violates these
assumptions.


## Pixel-origin repair after the 2026-09-14 blocked run

All five seeds completed MH repeat 0 and stopped at the first selected solve:
`Native render vs official policy pixels mismatch`. Neither repeat 1 nor
CEM-MH nor mechanism analysis had run. Missing `repeatability.json` means
**not_run**, not a measured repeatability failure.

The original check conflated two observation sources. In official
`World.evaluate_from_dataset`, the first solver input is the dataset frame;
subsequent inputs come from the live AddPixelsWrapper. That wrapper resizes
native RGB with PIL bilinear in uint8 BEFORE policy tensor conversion and
normalization. Native render -> normalize -> tensor resize is a different path.

The repaired diagnostic:

- keeps `restore_prefix`'s exact native render and full-physics checks;
- checks the first model input against the original dataset frame;
- checks later model inputs and real-rollout encoder inputs through the wrapper
  pixel path, then the policy's CHW `tv_tensors.Image` transform path;
- independently compares the reconstruction to the INSTALLED wrapper/policy;
- captures actual pre-transform policy pixels and goal frames in new traces;
- writes per-trace differences/shapes/hashes, and arrays on pixel-check failure;
- records the initial dataset/native image difference instead of erasing it;
- archives installed dataset-eval/wrapper/preprocessing source for diagnosis;
- disables video saving in diagnostic capture to avoid parallel output clashes.

No tolerance is loosened. No image is chosen by trying which alternative passes.
The source rule is fixed by the timestep. Goal tensors remain the exact dataset
goals supplied to the planner.

### Check saved MH traces before running more CEM

Run on the same server/environment, replacing the three directory arguments:

```bash
python eval_cem_mh_diagnostics.py \
  --stage check-saved --seed 42 \
  --input-dir OLD_RUN/input --reuse-mh-from OLD_RUN/seed42 \
  --output-dir NEW_RUN/preflight/seed42
```

This loads trusted server-local `.pt` traces created by the previous diagnostic,
checks dataset/config/checkpoint/input/manifest provenance, then checks every
saved MH solve input and returned-action replay. It does NOT run a new CEM or
compute a success comparison. Do not use untrusted `.pt` files.

When all five seed preflights pass, `--stage all --reuse-mh-from OLD_RUN/seed42`
in a NEW output directory reuses MH repeat 0, revalidates it, and executes the
remaining MH repeat 1, CEM-MH repeats 0/1 and gated mechanism analysis. Keep all
100 environments; do not run a selected-case-only world. Old output is read-only.

The uploaded results package excludes `.pt`; therefore the repair cannot be
claimed to pass the actual failed traces before this server-side preflight.
CPU regression tests cover dataset/live origin separation, wrapper-resize
semantics, raw/goal corruption, failure artifacts, legacy traces and provenance.

```bash
python -m unittest discover -s tests -p 'test_cem_mh_*.py' -v
```
