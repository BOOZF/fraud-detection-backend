"""The three analysis notebooks are executable documentation: each one must run top to bottom against the real
Teradata tables, and finish by passing its own checks. Run with:  pytest -m notebooks

Each notebook runs in a fresh `jupyter nbconvert --execute` process (importing nbclient inside pytest clashes with
the Teradata driver's C++ runtime), and the executed copy is read back as plain JSON."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parent.parent / "notebook"


def run(name: str, minutes: int, tmp_path: Path):
    done = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         f"--ExecutePreprocessor.timeout={minutes * 60}", "--output-dir", str(tmp_path), "--output", f"{name}.out",
         str(NB_DIR / f"{name}.ipynb")],
        capture_output=True, text=True, timeout=minutes * 60 + 120)
    assert done.returncode == 0, done.stderr[-2000:]
    nb = json.loads((tmp_path / f"{name}.out.ipynb").read_text())
    cells = nb["cells"]
    text = "\n".join(
        "".join(o["text"]) if o["output_type"] == "stream" else "".join(o.get("data", {}).get("text/plain", []))
        for c in cells if c["cell_type"] == "code" for o in c.get("outputs", []))
    return cells, text


@pytest.mark.notebooks
@pytest.mark.parametrize("name,minutes,expected", [
    ("Data_Understanding", 10, ["txn_scores", "fraud_xgb_model", "/api/kpis", "TD_VectorDistance"]),
    ("Data_Embedding", 15, ["text-embedding-3-small", "recall@3", "policy_emb"]),
    ("Model_Understanding", 25, ["train AUC", "test AUC", "VERDICT"]),
])
def test_notebook_runs_against_teradata_and_passes_its_own_checks(name, minutes, expected, tmp_path):
    cells, text = run(name, minutes, tmp_path)
    explanations = [c for c in cells if c["cell_type"] == "markdown"]
    assert len(explanations) >= 8, "a notebook meant for understanding needs step-by-step explanations"
    assert "ALL CHECKS PASSED" in text
    for needle in expected:
        assert needle in text, f"{needle!r} missing from {name} output"
    assert not any(o["output_type"] == "error" for c in cells if c["cell_type"] == "code" for o in c.get("outputs", []))
