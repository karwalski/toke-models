"""Story 132.77: cloud/post_train.sh must never commit adapter weights.

Runs the real script in a sandbox: a throwaway repo with the real .gitignore,
a stub `aws` that records what it was asked to upload, pushes and shutdown
disabled. Nothing touches AWS, Hugging Face or GitHub.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "cloud" / "post_train.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def sandbox(tmp_path: Path) -> dict:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    work = tmp_path / "work"
    adapter = work / "output" / "7b-qlora" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(os.urandom(4096))
    (adapter / "adapter_config.json").write_text("{}")
    (work / "output" / "7b-qlora" / "training_metrics.json").write_text('{"train_loss": 0.5}')
    (work / "train.log").write_text("log\n")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    _git(repo, "add", ".gitignore")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "aws_calls.txt"
    stub = bindir / "aws"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    stub.chmod(0o755)

    env = dict(os.environ)
    env.update(
        WORK_DIR=str(work),
        REPO_DIR=str(repo),
        MODEL_STORE_URI="s3://example-store/adapters",
        POST_TRAIN_PUSH="0",
        POST_TRAIN_SHUTDOWN="0",
        PATH=f"{bindir}{os.pathsep}{env['PATH']}",
    )
    return {"repo": repo, "env": env, "calls": calls}


def _run(sb: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], env=sb["env"], capture_output=True, check=False, text=True
    )


def test_weights_uploaded_not_committed(sandbox: dict) -> None:
    r = _run(sandbox)
    assert r.returncode == 0, r.stdout + r.stderr
    repo = sandbox["repo"]
    files = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert files, "post_train.sh made no commit"
    assert not [f for f in files if "tar" in f or f.endswith(".safetensors")], files
    ptr = [f for f in files if f.endswith("adapter-weights.pointer.json")]
    assert len(ptr) == 1, files
    # Every blob in the whole history is small: no weights reached git.
    for line in _git(repo, "rev-list", "--objects", "--all").splitlines():
        obj = line.split()[0]
        if _git(repo, "cat-file", "-t", obj).strip() == "blob":
            assert int(_git(repo, "cat-file", "-s", obj)) < 1024 * 1024
    pointer = json.loads(_git(repo, "show", f"HEAD:{ptr[0]}"))
    assert pointer["uri"].startswith("s3://example-store/adapters/train-results-")
    assert len(pointer["sha256"]) == 64 and pointer["bytes"] > 0
    calls = sandbox["calls"].read_text()
    assert "adapter-weights.tar.gz " + pointer["uri"] in calls


def test_no_store_means_no_commit(sandbox: dict) -> None:
    sandbox["env"]["MODEL_STORE_URI"] = ""
    head = _git(sandbox["repo"], "rev-parse", "HEAD")
    r = _run(sandbox)
    assert r.returncode != 0
    assert "MODEL_STORE_URI is unset" in r.stdout + r.stderr
    assert _git(sandbox["repo"], "rev-parse", "HEAD") == head
