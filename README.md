# Malaysia XX Bank Fraud Detection backend

Run (from this folder, port 8088):

    pip install -r requirements.txt
    python -m uvicorn service.app:app --host 0.0.0.0 --port 8088

Frontend (separate terminal): `cd ../fraud_detection_system && pnpm dev` (http://localhost:3000).

- `.env` (gitignored): `TD_HOST`, `TD_USER`, `TD_PASSWORD`, `OPENAI_API_KEY`.
- `scripts/`: `generate_data.py` -> `load_data.py` (Teradata `txn` + `txn_features`) -> `ingest_policies.py` (SOP embeddings).
- `notebook/Financial_Fraud_Detection_InDB_Python.ipynb`: trains the in-database XGBoost model and writes
  `fraud_xgb_model`, `txn_scores`, `model_metrics` (set `TD_PASSWORD` to run headless).
- `notebook/Financial_Fraud_Detection_AutoFraud_Python.ipynb`: AutoML comparison (not executed; resource heavy).
- Tests: `python -m pytest` (needs the Teradata tables above and `.env`; hits the real database and OpenAI).
