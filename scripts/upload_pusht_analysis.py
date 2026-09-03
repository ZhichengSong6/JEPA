#!/usr/bin/env python3
"""Upload small PushT diagnostic reports for analysis through the GitHub plugin.

Uses an isolated worktree and a fast-forward-only push. The original results,
current checkout, index and working edits are untouched. Large recordings and
candidate NPZs stay on the server. Requires only Python's standard library/git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid


REPORTS = (
    "summary.json", "paired_summary.json", "paired_manifest.csv",
    "lewm_closed_loop.json", "ald_tf_closed_loop.json", "cem_parity.json",
    "config.yaml", "run_identity.json", "provenance.json",
    "population_metrics.csv", "case_metrics.csv", "mean_execution_audit.csv",
    "skipped_solves.json",
)
PART_BYTES = 1024 * 1024
DEFAULT_BRANCH = "agent/stage1-bias-calibration"


def git(repo, *args, capture=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          text=True, stdout=subprocess.PIPE if capture else None)


def choose_run(repo, explicit=None):
    if explicit:
        run = Path(explicit).expanduser().resolve()
        if not run.is_dir():
            raise ValueError("Pass the original run DIRECTORY, not the large tar.gz file")
        return run
    candidates = []
    root = repo / "outputs" / "pusht_official_diagnostic"
    for path in root.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if summary.get("status") == "complete":
            candidates.append((summary.get("mode") == "formal", path.stat().st_mtime, path.parent))
    if not candidates:
        raise ValueError(f"No completed diagnostic under {root}; pass its run directory explicitly")
    return max(candidates, key=lambda item: item[:2])[2]


def utf8_parts(data, limit=PART_BYTES):
    """Bound tool-readable text blobs without changing any source bytes."""
    data.decode("utf-8")
    if limit < 4:
        raise ValueError("UTF-8 part size must be at least four bytes")
    if not data:
        yield b""
        return
    start = 0
    while start < len(data):
        end = min(start + limit, len(data))
        while end < len(data) and data[end] & 0xC0 == 0x80:
            end -= 1
        yield data[start:end]
        start = end


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare_reports(run, destination, limit=PART_BYTES):
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError(f"Diagnostic is not complete: {run}")
    missing = [name for name in REPORTS if not (run / name).is_file()]
    if missing:
        raise ValueError("Missing reports: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_run": run.name,
        "mode": summary.get("mode"),
        "official_benchmark": summary.get("official_benchmark"),
        "protocol": summary.get("protocol"),
        "paired_counts": summary.get("paired_counts"),
        "population_count": summary.get("num_populations"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transfer": "UTF-8 reports only; concatenate parts in order to recover each original file",
        "cleanup": "Remove this temporary directory after analysis; ordinary Git deletion retains history",
        "excluded": ["recordings/*.pt", "populations/*.npz", "checkpoints", "datasets", "videos"],
        "files": [],
    }
    for name in REPORTS:
        source = run / name
        if source.is_symlink():
            raise ValueError(f"Report must be a regular local file: {name}")
        data = source.read_bytes()
        parts = list(utf8_parts(data, limit))
        entry = {"source_name": name, "bytes": len(data), "sha256": sha256(data), "parts": []}
        for i, part in enumerate(parts):
            part_name = name if len(parts) == 1 else f"{name}.part{i + 1:03d}.txt"
            (destination / part_name).write_bytes(part)
            entry["parts"].append({"path": part_name, "bytes": len(part), "sha256": sha256(part)})
        manifest["files"].append(entry)
    manifest["total_report_bytes"] = sum(f["bytes"] for f in manifest["files"])
    (destination / "transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "README.md").write_text(
        f"# Temporary PushT analysis: {run.name}\n\n"
        "Start with `transfer_manifest.json`, `paired_summary.json`, `case_metrics.csv` "
        "and `mean_execution_audit.csv`. For split files, concatenate the listed parts "
        "as UTF-8 bytes and verify the original SHA-256 before parsing.\n\n"
        "This is a transfer of existing results, not a new experiment. "
        "Complete recordings and candidate arrays remain on the originating server. "
        "The owner requested deletion of this temporary directory after analysis.\n"
    )
    return manifest


def publish(repo, prepared, remote, branch):
    """Only the whitelisted inbox is committed; never stage the user's checkout."""
    git(repo, "check-ref-format", "--branch", branch)
    git(repo, "fetch", remote, branch, capture=False)
    parent = git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
    relative = Path("analysis_inbox") / prepared.name
    with tempfile.TemporaryDirectory(prefix="jepa-analysis-upload-") as scratch:
        checkout = Path(scratch) / "checkout"
        git(repo, "worktree", "add", "--detach", str(checkout), parent, capture=False)
        try:
            dest = checkout / relative
            if dest.exists():
                raise ValueError(f"Result directory already exists: {relative}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(prepared, dest)
            git(checkout, "add", "-f", "--", relative.as_posix())
            # Use the caller's existing identity when configured, without editing git config.
            identity = []
            for key, fallback in (("user.name", "JEPA analysis upload"), ("user.email", "analysis-upload@localhost")):
                result = subprocess.run(["git", "-C", str(repo), "config", "--get", key],
                                        text=True, stdout=subprocess.PIPE)
                if not result.stdout.strip():
                    identity += ["-c", f"{key}={fallback}"]
            git(checkout, *identity, "commit", "-m", f"Upload temporary PushT analysis: {prepared.name}", capture=False)
            commit = git(checkout, "rev-parse", "HEAD").stdout.strip()
            # A concurrent branch update causes a normal push failure; never force it.
            git(checkout, "push", remote, f"HEAD:refs/heads/{branch}", capture=False)
        finally:
            # Only our disposable worktree/copy is removed, even when push fails.
            git(repo, "worktree", "remove", "--force", str(checkout))
    return relative.as_posix(), commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Omit to select the newest completed formal run (or smoke if none)")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the compact text copy without GitHub writes")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run = choose_run(repo, args.run_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", run.name)
    prepared = repo / "outputs" / "github_analysis_upload" / f"{run_slug}_{stamp}_{uuid.uuid4().hex[:6]}"
    manifest = prepare_reports(run, prepared)
    print(f"Source: {run}", flush=True)
    print(f"Prepared: {prepared}", flush=True)
    print(f"Text reports: {manifest['total_report_bytes'] / 1024**2:.2f} MiB; no recordings/NPZs", flush=True)
    if args.prepare_only:
        return
    url = git(repo, "remote", "get-url", args.remote).stdout.strip()
    if not re.search(r"github\.com[:/]ZhichengSong6/JEPA(?:\.git)?/?$", url, re.I):
        raise ValueError("The selected remote must point to ZhichengSong6/JEPA on GitHub")
    relative, commit = publish(repo, prepared, args.remote, args.branch)
    print("\n=== UPLOAD COMPLETE ===")
    print(f"Branch: {args.branch}")
    print(f"Directory: {relative}")
    print(f"Commit: {commit}")
    print(f"Send this link: https://github.com/ZhichengSong6/JEPA/tree/{commit}/{relative}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Upload failed: {exc}\nOriginal results remain untouched; fix the error and rerun.")
