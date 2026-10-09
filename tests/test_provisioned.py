"""Acceptance for a freshly provisioned Teradata: the tables the old container had, and both policy PDFs
searchable and viewable through the API."""
from pathlib import Path

import pytest

from service import db

DOCS = Path(__file__).resolve().parent.parent / "docs"
REQUIRED_TABLES = {"txn", "txn_scores", "fraud_xgb_model", "model_metrics", "policy_docs", "policy_chunks",
                   "policy_emb", "q_emb"}
REQUIRED_VIEWS = {"txn_features"}


def _names(*kinds: str) -> set[str]:
    in_list = ", ".join(f"'{k}'" for k in kinds)
    rows = db.query(f"SELECT TableName FROM DBC.TablesV WHERE DatabaseName = DATABASE AND TableKind IN ({in_list})")
    return {r[0].strip().lower() for r in rows}


def test_all_tables_exist():
    assert REQUIRED_TABLES <= _names("T", "O")  # O = NoPI tables (teradataml writes the model tables without a primary index)


def test_feature_view_exists():
    assert REQUIRED_VIEWS <= _names("V")


def test_every_transaction_is_scored():
    (txn, scored), = db.query("SELECT (SELECT COUNT(*) FROM txn), (SELECT COUNT(*) FROM txn_scores)")
    assert txn > 0 and txn == scored


def test_model_metrics_recorded(client):
    m = client.get("/api/model").json()
    assert "XGBoost" in str(m) and 0.5 < float(m.get("auc", 0)) <= 1


@pytest.mark.parametrize("pdf", sorted(p.name for p in DOCS.glob("*.pdf")))
def test_pdf_is_listed_and_opens_as_pdf(client, pdf):
    listed = {d["doc"] for d in client.get("/api/documents").json()}
    assert pdf in listed
    r = client.get(f"/api/documents/{pdf}/file")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
