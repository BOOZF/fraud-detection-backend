# Malaysia XX Bank: fraud detection backend

FastAPI service on port 8088. Everything analytical runs **inside Teradata**: the XGBoost model is trained and scored there,
the transactions and scores live there, and the policy PDF is searched there (`TD_VectorDistance`). Python orchestrates.

## Run

    pip install -r requirements.txt
    python -m uvicorn service.app:app --host 0.0.0.0 --port 8088      # from this folder

Frontend (separate terminal): `cd ../fraud_detection_system && pnpm dev`  ->  http://localhost:3000

Configuration is in `.env` (gitignored; copy `.env.example`): `TD_HOST`, `TD_USER`, `TD_PASSWORD`, `OPENAI_API_KEY`, and the models:

| Setting | Default | Controls |
|---|---|---|
| `LLM_MODEL` | `gpt-5-mini` | **Every** LLM call: chat answers, the alert brief, the relevance check on brief answers, the on-topic gate |
| `REASONING_EFFORT` | `low` | gpt-5 models only (`minimal`/`low`/`medium`/`high`) |
| `EMBED_MODEL` | `text-embedding-3-small` | Document and question embeddings. After changing it run `python scripts/ingest_policies.py` |
| `VERIFY_MODEL`, `GATE_MODEL` | blank | Optional overrides; blank means use `LLM_MODEL` |
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

## Copilot behaviour

* **Citations are highlighted.** `GET /api/documents/{name}/file?chunk=<id>` returns the PDF with the cited passage highlighted as real
  PDF annotations (`service/services/highlight.py`, PyMuPDF); the viewer opens it at the cited page.
* **Briefs never change.** A brief is generated once (temperature 0, fixed seed), stored in Teradata (`copilot_briefs`) and read back; it is
  regenerated only when the document set changes. Shape: a short verdict + up to 3 bullet points per question.
* **Chat guardrails** (`service/guardrails.py`): prompt-injection / secret screening, PII masking in and out, a topic gate (fraud at this bank only,
  with a document-similarity second opinion), and a fixed refusal. Overview questions answer as a table; questions about one alert as prose.

## Chat answers have one fixed shape

The chat model returns a small JSON schema (`kind`, `summary`, `sections`, `table`, `takeaway`); `service/services/structured.py` renders it to Markdown
in a fixed order. `alert` answers always have **Key facts**, **Why it was flagged** (both filled from Teradata by the code, never by the model),
**Policy guidance**, **Suggested next steps**; `data` answers are a table plus one sentence; `policy` answers are **What the policy says** /
**How it applies**; anything else is one sentence. A reply that is not valid JSON is asked for once more.

**Citations.** Each cited page shows as a card with the cited paragraph and the sentence the answer relies on marked; the `[document p.N]`
references inside the answer become small page chips. Opening one serves the PDF with the paragraph highlighted in yellow and the key sentence in
orange (`GET /api/documents/{name}/file?chunk=<id>&quote=<sentence>`). Brief answers use their verbatim evidence sentence as the key sentence; chat
answers use the sentence of the cited chunk that shares the most keywords with the points that cite it.

**Streaming.** `POST /api/chat/stream` (same body as `/api/chat`) answers as server-sent events: `step` (what the copilot is doing: checking the
question, reading the alert, each Teradata lookup, each policy search with the passages it found), `answer` (the answer so far, already in the fixed
layout; replace, don't append), then `done` (answer, citations, tools, guardrail) or `error`. The chat panel shows the latest step while it works,
with a button to expand all of them, and keeps them afterwards under "How this was answered". gpt-5 models do not expose their private reasoning, so
the steps are the real actions the service took, not a transcript of the model's thoughts.

## Is the RAG answering correctly? Measure it

Two benchmarks (use the real OpenAI key; a few cents each) that separate *retrieval* problems from *answer* problems:

    python scripts/eval_retrieval.py      # does the right chunk come back? hit@k / MRR for embedding models, with and without the keyword re-rank
    python scripts/eval_generation.py     # are the brief's answers faithful to the excerpts? a stronger judge model audits every answer

Findings on the current documents (Oct 2026): retrieval is fine (source chunk in the top 3 about 90% of the time, 93% on the right page;
text-embedding-3-large would add about 5 points). The weak spot is that `Fraud_Detection_SOP.pdf` is a USCIS immigration SOP and the second
PDF is a university policy, so most card-fraud questions (urgency, next step, regulator report, customer contact) are simply not answered
in them. The brief therefore: builds "Why flagged" and the headline from the rule-based reason codes, makes the model quote the sentence it
relies on, checks the quote is really in the cited chunk, has an independent model confirm the passage answers the question for a *card*
alert, and otherwise shows **Not covered** with no citation. All of it runs on the one `LLM_MODEL` (the benchmark showed a stronger, separate relevance-check
model accepted slightly fewer misapplied quotes, 95% vs 92% on 40 cases; set `VERIFY_MODEL` to try it). To get real answers, upload the bank's own card-fraud procedure on the Documents page.

## API

`/api/health`, `/api/kpis`, `/api/overview` (daily fraud/alerts/transactions per channel and fraud by merchant, for the dashboard charts), `/api/model`, `/api/alerts` (all alerts, highest probability first), `/api/alerts/{id}` (+ customer),
`POST /api/alerts/{id}/brief` (structured copilot answers with page citations), `POST /api/score` (re-score in Teradata),
`POST /api/chat` and `POST /api/chat/stream` (data-aware chat; optional `alert_id` / `context` for a highlighted alert),
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
