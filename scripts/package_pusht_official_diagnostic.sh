#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash $0 /path/to/run_directory" >&2
  exit 2
fi
python - "$1" <<'PY'
import json
import sys
import tarfile
from pathlib import Path

run = Path(sys.argv[1]).resolve()
required = ["summary.json", "paired_summary.json", "paired_manifest.csv",
            "lewm_closed_loop.json", "ald_tf_closed_loop.json", "cem_parity.json",
            "config.yaml", "run_identity.json", "provenance.json",
            "population_metrics.csv", "case_metrics.csv", "mean_execution_audit.csv",
            "skipped_solves.json"]
for name in required:
    if not (run / name).is_file():
        raise SystemExit(f"Missing result: {run / name}")
summary = json.loads((run / "summary.json").read_text())
if summary.get("status") != "complete":
    raise SystemExit("Diagnostic is not complete")
files = sorted((run / "populations").glob("*.npz"))
if len(files) != summary["num_populations"]:
    raise SystemExit("Population count differs from summary")
dest = run.with_name(run.name + "_bundle.tar.gz")
tmp = dest.with_name(dest.name + ".tmp")
with tarfile.open(tmp, "w:gz") as tar:
    for path in [run / name for name in required] + files:
        tar.add(path, arcname=str(Path(run.name) / path.relative_to(run)))
tmp.replace(dest)
print(f"Upload: {dest}")
print("Full replay recordings stay in the run directory and are not included in the bundle.")
PY
