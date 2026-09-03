# Official PushT paired diagnostic

Run LeWM and ALD+TF on the **same 100 dataset starts**, seed 42, using
`N=300, I=30, K=30`, horizon/action block/receding horizon 5, evaluation budget
50 and one real observation. Do not use the previous B=3000 case IDs or expected
success rates. The current branch is `agent/stage1-bias-calibration`.

```bash
conda activate lewm
NODE=4090node3 bash scripts/submit_pusht_official_diagnostic.sh smoke
# Once the smoke finishes with === SLURM DONE ===:
NODE=4090node3 \
FACTOR_POLICY=pusht_lewm_factor_seed0_ep10/lewm_factor_epoch_10 \
bash scripts/submit_pusht_official_diagnostic.sh formal
```

`FACTOR_POLICY` is optional. With it, the existing Factor model scores the exact
same observed futures and candidate actions after the benchmark. It is not
trained or used to select the official trajectories. There is no new Factor
closed-loop success number in this run. Include it in the smoke too to check
that checkpoint. Omitting it runs only LeWM and ALD+TF. The checkpoint names are
relative to `STABLEWM_HOME` and omit `_object.ckpt`.

The runner uses one GPU on 4090node3 by default. `REPO`, `CONDA_SH`, `CONDA_ENV`,
`STABLEWM_HOME`, `NODE`, and checkpoint policies can be overridden. `DRY_RUN=1`
generates and syntax-checks the Slurm file without submission. Every new run
gets a unique directory; no previous results are deleted. A single
`<run_directory>_bundle.tar.gz` is produced automatically. Upload that bundle.

## What is measured

The exact installed `stable-worldmodel==0.0.6` CEM is checked against the existing
tracing implementation using a deterministic cost with and without a warm start.
An action/cost mismatch stops the job before the benchmark. Training code,
inference cost, success criterion and official policy execution are unchanged.

Both models execute once. All non-both-success episodes plus up to three
difficulty-matched both-success controls are chosen from this run. For each
source trajectory, solve 0 and solve 1 populations are replayed at CEM iterations
0, 3, 9, 19 and 29. Missing solves (for example after early success) are reported.
When both policies succeed everywhere, the hardest initial success cases are
used as controls instead. Smoke uses six starts and a small search only to test
the code path; it is explicitly not the official benchmark.

| Quantity | Meaning |
| --- | --- |
| `encoder_raw` | Squared Euclidean distance between encoded real candidate future and encoded goal |
| `predictor_raw` | The corresponding predicted terminal latent distance; audited against the native source CEM cost |
| `endpoint_mse_mean` | Predictor terminal latent error relative to the real future encoding |
| `endpoint_nmse_population` | Endpoint MSE divided by encoded population variance; undefined for collapsed populations |
| `pred_encoder_rho` | Whether predictor and encoder order the same candidates consistently |
| `physical_*` | Ranking against the normalized physical diagnostic cost, plus elite overlap/selection regret |
| `pusher/block/theta_*` | Per-factor ranking, partial Spearman and matched-factor accuracy |
| `*_matched_pairs` | Number of eligible pairs; pairs are correlated, not independent trials |
| `*_informative` | Whether the factor has enough IQR to interpret the ranking |
| `encoder_factor_head_diagnostic_only` | Optional decoded factor distance, never used by CEM |

Physical cost is `(pusher_error/20)^2 + (block_error/20)^2 +
(wrapped_theta_error/(pi/9))^2`. It is a continuous diagnostic surrogate.
Benchmark success remains the environment's official success signal. Pusher
position is part of that task. Object-only cost is a diagnostic decomposition.

Partial correlations and matched costs are observational checks. They do not
prove a causal encoder collapse or absence of information. In the new matched
tests, **both** other factor costs are approximately fixed; the old
block-matched theta test did not also fix pusher cost. Therefore these stricter
matched accuracies should not be compared numerically as if the old test were
identical. Constant/near-constant target IQR returns an uninformative result.

Candidate physics uses state resets that may omit hidden simulator state.
`mean_execution_audit.csv` checks the actual executed actions against the final
CEM mean and reports replay endpoint mismatch where the actual endpoint was
recorded. The actual trajectory is never reconstructed from state resets.
Read this audit before attributing replay discrepancies to the world model.

## Factor-aware interpretation

`train.py` applies the absolute and metric auxiliary losses to three nonlinear
factor heads. The metric term compares **head-output distances**, not raw
latent distances. Its weights are 0.10 for encoded factors, 0.05 for predicted
factors and 0.05 for the metric term. Pusher/block coordinates and sin/cos of
orientation are supervised during training. `jepa.py:criterion` still uses
terminal raw-latent squared distance during CEM.

Better decoded factors therefore do not guarantee better Euclidean latent goal
geometry. The optional Factor comparison separates this distinction on exactly
the same states and actions. If Factor improves raw encoder object/theta ranking
but its predictor does not follow, the old representation may be worth combining
with predictor calibration. If only head-based geometry improves, it has not yet
repaired the distance used by CEM. No training decision is made by this script.

The current ALD+TF recipe starts from `lewm_epoch_10` and freezes encoder and
projector. It does not inherit the earlier Factor encoder. Visual parameter
fingerprints are reported to check this on the actual loaded checkpoints.
Raw endpoint MSE is scale-dependent across different encoders; use physical
ranking and the normalized endpoint error for the Factor comparison.

## Results and replay recovery

`paired_summary.json` reports counts, paired percentage-point difference and an
exact two-sided McNemar p-value; a single seed/100 starts is still limited evidence.
`population_metrics.csv` has every scored population. `case_metrics.csv` averages
within episode/source/solve/model first, so iterations are not mistaken for
independent cases. `populations/*.npz` contains actual normalized candidates,
future physical states, costs and endpoint errors. `provenance.json` and
`run_identity.json` identify the code/environment and exact starts.

Large complete run recordings stay on the server, outside the upload bundle.
If replay fails **after both closed-loop recordings have been saved**, fix the
cause and resume with the same mode, policies and directory:

```bash
STAGE=replay RUN_DIR=/absolute/path/printed/by/the/original/job \
NODE=4090node3 \
FACTOR_POLICY=pusht_lewm_factor_seed0_ep10/lewm_factor_epoch_10 \
bash scripts/submit_pusht_official_diagnostic.sh formal
```

This redoes the offline diagnostic using saved runs; it does not rerun CEM or
change case labels. Completed results are not overwritten.

## Upload results through GitHub when the bundle is too large

```bash
python scripts/upload_pusht_analysis.py
```

The uploader automatically selects the newest completed formal run, or the
newest complete smoke if there is no formal result. To choose a particular run,
pass the original run directory as its positional argument. No experiment is
rerun and no GPU is needed. `--prepare-only` creates the compact copy locally.

Only the 13 JSON/CSV/YAML reports are copied to `analysis_inbox/<unique-run-id>`
on `agent/stage1-bias-calibration`. Large recordings and candidate arrays stay
on the server. Reports larger than 1 MiB are split into UTF-8 text parts, with
byte counts and SHA-256 hashes in `transfer_manifest.json`. Reassemble parts
in manifest order before parsing CSV/JSON. These reports are sufficient for
the first comparison of paired outcomes, Factor geometry, theta ranking and
replay reliability; individual candidate arrays can be requested later if needed.

The upload uses an isolated disposable worktree, existing GitHub push credentials
and a normal fast-forward-only push. It does not stage current work, change the
current branch, overwrite remote changes or delete source results. Send the
printed GitHub link after `=== UPLOAD COMPLETE ===` appears. After analysis,
remove only this run's temporary inbox directory. Normal deletion removes files
from the branch's current contents; Git commit history retains them.
