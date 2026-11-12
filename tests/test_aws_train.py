"""Story 132.82: cloud/aws_train.sh uploads the checked-in post-train script,
and no failure can leave a GPU instance running silently.

Runs the real cloud/aws_train.sh end to end against stubs. Nothing touches
AWS, GitHub or a network:

- `aws`, `scp`, `ssh`, `tmux`, `sudo`, `gh`, `nvidia-smi` and the training
  step (`python3 finetune/train_qlora.py`) are shell stubs that log what they
  were asked to do;
- `ssh` is a fake remote: it runs the command locally, with a clean
  environment, after mapping /home/ubuntu to a temporary directory in the
  command and in what it is sent on stdin (`scp` maps its destination the
  same way); the setup step (pip installs) is logged and skipped;
- `tmux new-session` runs its command synchronously.

Scenario knobs are files in ctl/, not environment variables, so nothing the
test sets can leak into the "remote" side except through aws_train.sh itself.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLOUD = ROOT / "cloud"

STUBS = {
    # Canned answers for everything aws_train.sh asks of AWS.
    "aws": r"""#!/bin/sh
echo "$*" >> "$T/log/aws"
case "$1 $2" in
  "ec2 describe-images") echo ami-test ;;
  "ec2 describe-security-groups") echo sg-test ;;
  "ec2 request-spot-instances") echo sir-test ;;
  "ec2 describe-spot-instance-requests") echo i-test ;;
  "ec2 describe-instances") echo 192.0.2.10 ;;
  "ec2 terminate-instances") echo shutting-down ;;
  "s3 cp") [ -f "$T/ctl/s3_fail" ] && { echo "upload failed" >&2; exit 1; } ;;
esac
exit 0
""",
    "scp": r"""#!/bin/sh
[ -f "$T/ctl/scp_fail" ] && { echo "scp: connection reset" >&2; exit 1; }
for a; do src="$dst"; dst="$a"; done
dst="${dst#*:}"
dst="$(printf %s "$dst" | sed "s#^/home/ubuntu#$T/remote#")"
echo "$src -> $dst" >> "$T/log/scp"
case "$dst" in */) dst="$dst$(basename "$src")" ;; esac
[ -d "$dst" ] && dst="$dst/$(basename "$src")"
cp "$src" "$dst"
""",
    "ssh": r"""#!/bin/bash
for a; do cmd="$a"; done
map() { sed "s#/home/ubuntu#$T/remote#g"; }
cmd="$(printf %s "$cmd" | map)"
n=$(ls "$T/log" | grep -c '^ssh\.' || true)
remote() { env -i HOME="$T/remote" T="$T" PATH="$T/remote-bin:/usr/bin:/bin:/usr/sbin:/sbin" bash "$@"; }
if [ "$cmd" = "bash -s" ]; then
  script="$(map)"
  printf %s "$script" > "$T/log/ssh.$n.script"
  case "$script" in
    *"pip install"*) exit 0 ;;          # setup_remote: skipped
    *) remote -c "$script" ;;
  esac
else
  printf %s "$cmd" > "$T/log/ssh.$n.cmd"
  map | remote -c "$cmd"
fi
""",
}

REMOTE_STUBS = {
    "tmux": r"""#!/bin/bash
[ "$1" = new-session ] || exit 0
shift
while [ "$1" = -d ] || [ "$1" = -s ]; do [ "$1" = -s ] && shift; shift; done
"$@" > "$T/log/tmux.out" 2>&1
echo $? > "$T/log/tmux.exit"
exit 0
""",
    "sudo": r"""#!/bin/sh
echo "$*" >> "$T/log/sudo"
""",
    "gh": r"""#!/bin/sh
echo "$*" >> "$T/log/gh"
""",
    "nvidia-smi": "#!/bin/sh\necho 'FakeGPU, 24576 MiB'\n",
    # No network from the fake remote: the old embedded script asked the EC2
    # metadata endpoint for the instance type, which hangs off EC2.
    "curl": '#!/bin/sh\necho "$*" >> "$T/log/curl"\nexit 7\n',
    "wget": '#!/bin/sh\necho "$*" >> "$T/log/wget"\nexit 4\n',
    "aws": STUBS["aws"],
}


def _python_stub(real: str) -> str:
    return f"""#!/bin/sh
if [ "$1" = finetune/train_qlora.py ]; then
  echo "fake training $*"
  [ -f "$T/ctl/train_fail" ] && {{ echo "CUDA out of memory" >&2; exit 1; }}
  mkdir -p output/7b-qlora/adapter
  echo weights > output/7b-qlora/adapter/adapter_model.safetensors
  echo '{{"train_loss": 0.5}}' > output/7b-qlora/training_metrics.json
  exit 0
fi
exec "{real}" "$@"
"""


def _write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def sb(tmp_path: Path) -> dict:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    t = tmp_path
    for d in ("log", "ctl", "bin", "remote-bin", "remote/toke-venv/bin", "remote/.ssh"):
        (t / d).mkdir(parents=True)
    (t / "remote/toke-venv/bin/activate").write_text("")

    # A copy of the project, as aws_train.sh finds it relative to itself.
    proj = t / "proj"
    shutil.copytree(CLOUD, proj / "cloud", ignore=shutil.ignore_patterns(".train_state.json"))
    for rel in ("finetune/train_qlora.py", "finetune/merge_adapters.py",
                "finetune/configs/7b.yaml", "training-data/train.jsonl",
                "training-data/eval.jsonl"):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text("stub\n")

    # The results repo post_train.sh commits into, pre-cloned so no clone runs.
    repo = t / "remote/toke-model-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    _git(repo, "add", ".gitignore")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")

    for name, text in STUBS.items():
        _write_exec(t / "bin" / name, text)
    for name, text in REMOTE_STUBS.items():
        _write_exec(t / "remote-bin" / name, text)
    _write_exec(t / "remote-bin" / "python3", _python_stub(sys.executable))
    key = t / "key.pem"
    key.write_text("not a key\n")

    env = {
        "PATH": f"{t / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(t / "home"),
        "T": str(t),
        "KEY_FILE": str(key),
        "GITHUB_TOKEN": "test-token",
        "MODEL_STORE_URI": "s3://example-store/adapters",
        "POST_TRAIN_PUSH": "0",
    }
    return {"t": t, "proj": proj, "env": env}


def _run(sb: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(sb["proj"] / "cloud" / "aws_train.sh"), *args],
        env=sb["env"], capture_output=True, check=False, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )


def _log(sb: dict, name: str) -> str:
    p = sb["t"] / "log" / name
    return p.read_text() if p.exists() else ""


def _shutdowns(sb: dict) -> list[str]:
    return [" ".join(ln.split()[:3]) for ln in _log(sb, "sudo").splitlines()
            if ln.startswith("shutdown")]


def _ctl(sb: dict, name: str) -> None:
    (sb["t"] / "ctl" / name).write_text("1")


def _out(r: subprocess.CompletedProcess, sb: dict) -> str:
    return r.stdout + r.stderr + _log(sb, "tmux.out")


def test_uploads_the_checked_in_post_train_script(sb: dict) -> None:
    r = _run(sb)
    assert r.returncode == 0, _out(r, sb)
    on_instance = sb["t"] / "remote/toke-model/cloud/post_train.sh"
    assert on_instance.read_bytes() == (CLOUD / "post_train.sh").read_bytes()
    assert "release" not in _log(sb, "gh"), "published a GitHub Release"
    s3 = [ln for ln in _log(sb, "aws").splitlines() if ln.startswith("s3 cp")]
    assert any("adapter-weights.tar.gz s3://example-store/adapters/" in ln for ln in s3), s3
    ptr = _git(sb["t"] / "remote/toke-model-repo", "show", "--name-only", "--format=", "HEAD")
    assert "adapter-weights.pointer.json" in ptr
    # The token travels on stdin, never on a command line, and is gone after.
    for f in (sb["t"] / "log").glob("ssh.*"):
        assert "test-token" not in f.read_text(), f.name
    assert not (sb["t"] / "remote/.git-credentials").exists()


def test_success_arms_the_dead_mans_switch_then_shuts_down(sb: dict) -> None:
    r = _run(sb)
    assert r.returncode == 0, _out(r, sb)
    assert _shutdowns(sb) == ["shutdown -h +720", "shutdown -h +1"], _log(sb, "sudo")


def test_training_failure_shuts_the_instance_down(sb: dict) -> None:
    _ctl(sb, "train_fail")
    r = _run(sb)
    assert r.returncode == 0, _out(r, sb)  # the launch itself succeeded
    assert _shutdowns(sb) == ["shutdown -h +720", "shutdown -h +30"], _log(sb, "sudo")


def test_post_train_failure_shuts_the_instance_down(sb: dict) -> None:
    _ctl(sb, "s3_fail")
    r = _run(sb)
    assert r.returncode == 0, _out(r, sb)
    assert _shutdowns(sb) == ["shutdown -h +720", "shutdown -h +30"], _log(sb, "sudo")


def test_keep_alive_opt_out_is_explicit_and_loud(sb: dict) -> None:
    _ctl(sb, "train_fail")
    sb["env"]["KEEP_ALIVE_ON_FAILURE"] = "1"
    r = _run(sb)
    assert r.returncode == 0, _out(r, sb)
    assert _shutdowns(sb) == ["shutdown -h +720"], _log(sb, "sudo")
    assert "stays UP" in _out(r, sb)


def test_local_failure_after_provisioning_terminates_the_instance(sb: dict) -> None:
    _ctl(sb, "scp_fail")
    r = _run(sb)
    assert r.returncode != 0
    assert "ec2 terminate-instances --region us-east-1 --instance-ids i-test" in _log(sb, "aws"), _log(sb, "aws")


def test_local_failure_opt_out_keeps_the_instance_and_says_so(sb: dict) -> None:
    _ctl(sb, "scp_fail")
    sb["env"]["KEEP_INSTANCE_ON_ERROR"] = "1"
    r = _run(sb)
    assert r.returncode != 0
    assert "terminate-instances" not in _log(sb, "aws")
    assert "i-test" in r.stderr and "still running" in r.stderr, r.stderr


def test_model_store_is_required_before_any_instance_is_requested(sb: dict) -> None:
    del sb["env"]["MODEL_STORE_URI"]
    r = _run(sb)
    assert r.returncode != 0
    assert "MODEL_STORE_URI" in r.stderr
    assert "request-spot-instances" not in _log(sb, "aws")
