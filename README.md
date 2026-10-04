# Malaysia XX Bank: fraud detection backend

FastAPI service on port 8088. Everything analytical runs **inside Teradata**: the XGBoost model is trained and scored there,
the transactions and scores live there, and the policy PDF is searched there (`TD_VectorDistance`). Python orchestrates.

## Run

    pip install -r requirements.txt
    python -m uvicorn service.app:app --host 0.0.0.0 --port 8088      # from this folder

Frontend (separate terminal): `cd ../fraud_detection_system && pnpm dev`  ->  http://localhost:3000

Configuration is in `.env` (gitignored): `TD_HOST`, `TD_USER`, `TD_PASSWORD`, `OPENAI_API_KEY`.
The service takes about 30 s to start (it logs on to Teradata and warms its caches).

## Understand it: three notebooks (run them top to bottom; every number is queried live)

| Notebook | Answers |
|---|---|
| `notebook/Data_Understanding.ipynb` | Which tables exist in Teradata, which data is used, where the model and the scores live, and the **real SQL behind every UI element** |
| `notebook/Data_Embedding.ipynb` | How the PDF is extracted, cleaned, chunked, embedded and searched, and measurable proof that it is embedded well |
| `notebook/Model_Understanding.ipynb` | How the model was trained, and whether AUC 0.92 is overfitting or underfitting (it is not overfitting; mildly underfitting; see the verdict) |

`notebook/Financial_Fraud_Detection_InDB_Python.ipynb` is the training notebook that writes `fraud_xgb_model`, `txn_scores` and `model_metrics`.

## Knowledge base

The copilot answers from the PDF(s) in `docs/` (currently `Fraud_Detection_SOP.pdf`) and from anything uploaded on the Documents page.
`python scripts/ingest_policies.py` rebuilds it from `docs/` (reset + ingest). Cleaning, chunking (800 chars) and hybrid search
(vector candidates from Teradata, re-ranked by keyword overlap) are in `service/services/documents.py` and `service/services/rag.py`.

## API

`/api/health`, `/api/kpis`, `/api/model`, `/api/alerts` (all alerts, highest probability first), `/api/alerts/{id}` (+ customer),
`POST /api/alerts/{id}/brief` (structured copilot answers with page citations), `POST /api/score` (re-score in Teradata),
`POST /api/chat` (data-aware chat; optional `alert_id` / `context` for a highlighted alert),
`/api/documents` (list, upload, delete, `/{name}/file` returns the original).

## Tests (they use the real Teradata tables and OpenAI)

    python -m pytest                      # ~9 minutes: API, retrieval, brief, chat, documents, resilience
    python -m pytest tests/test_pdf_cleaning.py     # seconds, no database
    python -m pytest -m notebooks         # ~20 minutes: executes the three notebooks end to end

## Things worth knowing

* teradataml shares ONE connection per process and it is not safe for simultaneous use. `service/db.py` serialises access with a lock
  and reconnects once if the connection dies (a reconnect takes ~20 s). Read results are DataFrames built from `execute_sql` rows.
* Teradata XGBoost needs `num_boosted_trees` >= the number of AMPs holding data (4 here). The model is sensitive to its settings
  (see `Model_Understanding.ipynb`); keep the recipe in `scripts/train_model.py` fixed unless you re-validate.
* Scores rank transactions; they are not calibrated probabilities. At the 0.8 alert threshold precision is ~60% but recall is only ~6%.
