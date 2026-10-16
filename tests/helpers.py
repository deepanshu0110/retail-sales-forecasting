import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args, timeout=900):
    """Run a project script from the repo root and fail the test if it exits non-zero."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{' '.join(args)} failed:\n{output[-3000:]}"
    assert "Traceback" not in output, f"{' '.join(args)} printed a traceback:\n{output[-3000:]}"
    return output
