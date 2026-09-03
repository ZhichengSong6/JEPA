import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "upload_pusht_analysis.py"
spec = importlib.util.spec_from_file_location("upload_pusht_analysis", MODULE)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


class UploadTests(unittest.TestCase):
    def make_run(self, root, name="formal_test", mode="formal"):
        run = root / name
        run.mkdir(parents=True)
        for report in upload.REPORTS:
            (run / report).write_text('{"note":"角度🙂"}\n')
        (run / "summary.json").write_text(json.dumps({"status": "complete", "mode": mode}))
        return run

    def test_report_whitelist_and_lossless_utf8_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_run(root)
            (run / "recordings").mkdir()
            (run / "recordings" / "large.pt").write_bytes(b"not for upload")
            dest = root / "prepared"
            manifest = upload.prepare_reports(run, dest, limit=17)
            self.assertEqual(len(manifest["files"]), len(upload.REPORTS))
            for entry in manifest["files"]:
                parts = [(dest / part["path"]).read_bytes() for part in entry["parts"]]
                for part in parts:
                    part.decode("utf-8")
                    self.assertLessEqual(len(part), 17)
                self.assertEqual(b"".join(parts), (run / entry["source_name"]).read_bytes())
                self.assertEqual(upload.sha256(b"".join(parts)), entry["sha256"])
            self.assertFalse((dest / "recordings").exists())

    def test_latest_formal_preferred_over_smoke_and_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / "outputs" / "pusht_official_diagnostic"
            formal = self.make_run(root)
            self.make_run(root, "smoke_new", "smoke")
            incomplete = self.make_run(root, "formal_incomplete")
            (incomplete / "summary.json").write_text('{"status":"running","mode":"formal"}')
            self.assertEqual(upload.choose_run(repo), formal)

    def test_publish_preserves_checkout_and_staged_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, remote = root / "repo", root / "remote.git"
            repo.mkdir()
            git(root, "init", "--bare", str(remote))
            git(repo, "init", "-b", "analysis-test")
            git(repo, "config", "user.name", "Test")
            git(repo, "config", "user.email", "test@localhost")
            (repo / "tracked.txt").write_text("original\n")
            git(repo, "add", "tracked.txt")
            git(repo, "commit", "-m", "base")
            git(repo, "remote", "add", "origin", str(remote))
            git(repo, "push", "origin", "HEAD")
            # Neither a staged edit nor an untracked file may enter the upload commit.
            (repo / "tracked.txt").write_text("user edit\n")
            git(repo, "add", "tracked.txt")
            (repo / "unrelated.txt").write_text("local only\n")
            before_head = git(repo, "rev-parse", "HEAD")
            before_status = git(repo, "status", "--porcelain")
            before_index = git(repo, "diff", "--cached")
            run = self.make_run(root)
            prepared = root / "prepared"
            upload.prepare_reports(run, prepared)
            relative, commit = upload.publish(repo, prepared, "origin", "analysis-test")
            self.assertEqual(git(repo, "rev-parse", "HEAD"), before_head)
            self.assertEqual(git(repo, "status", "--porcelain"), before_status)
            self.assertEqual(git(repo, "diff", "--cached"), before_index)
            self.assertEqual(git(remote, "show", f"{commit}:tracked.txt"), "original")
            changed = git(remote, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
            self.assertTrue(all(name.startswith(relative + "/") for name in changed))
            self.assertEqual(git(remote, "rev-parse", "analysis-test"), commit)
            self.assertEqual(git(repo, "worktree", "list", "--porcelain").count("worktree "), 1)
            self.assertTrue((run / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
