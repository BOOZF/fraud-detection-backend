"""General-purpose copilot chat. The model can only call a fixed set of read-only tools (no SQL from the
model), so numbers always come from Teradata and policy answers from the uploaded documents."""
import json
import re

from .. import data, db, reasons
from ..services import llm, rag

CHANNELS = ["CARD_POS", "CARD_ECOM", "DUITNOW", "FPX", "ATM"]
MERCHANTS = ["GROCERY", "FNB", "ELECTRONICS", "TRAVEL", "LUXURY", "CRYPTO", "GAMING"]
GROUPS = {"channel": "t.channel", "merchant_cat": "t.merchant_cat", "hour_of_day": "t.hour_of_day",
          "priority": "CASE WHEN s.Prob_1 >= 0.9 THEN 'P1' WHEN s.Prob_1 >= 0.8 THEN 'P2' ELSE 'below alert threshold' END"}
MAX_TOOL_ROUNDS = 5

SYSTEM = (
    "You are the fraud-operations copilot for Malaysia XX Bank, shown as a chat bubble inside the fraud dashboard. "
    "You answer two kinds of question: (1) about the bank's scored transactions and alerts, and (2) about the uploaded "
    "policy documents. "
    "Definitions: an ALERT is a transaction whose model fraud probability is at least 0.80 (80%); Priority 1 (P1) is "
    "0.90 or above, Priority 2 (P2) is 0.80 to 0.89. These are model-score bands only; response deadlines, if any, "
    "come from the policy documents, never from your memory. "
    "ALWAYS use the tools for any number, transaction or policy fact; never guess or invent figures. For questions "
    "about alerts as a whole use alert_stats (counts, the P1/P2 split, amounts), alerts_breakdown (grouped) or list_alerts (individual alerts). Never add or subtract counts yourself; ask the tool for the figure. For questions about the "
    "uploaded policies use search_policies, answer only from the excerpts it returns, and cite them as "
    "[document p.N] using the source label. Any question that is not about the scored transactions is a question "
    "about the uploaded documents, even if its topic does not look related to banking: call search_policies "
    "FIRST, and only say you cannot answer if the excerpts contain nothing relevant. The uploaded policy may have been written for a different organisation "
    "or type of fraud than this bank's card fraud: report what it says, and mention briefly when it only applies "
    "by analogy. "
    "Quote numbers exactly as the tools return them. Keep answers short and concrete (a few sentences or a short "
    "list). If a FOCUSED ALERT block is present, 'this alert', 'it' and 'this transaction' mean that alert: answer "
    "from its facts, and use tools for anything beyond them. The user may also attach text they highlighted on screen "
    "as 'Selected text'; treat it as the subject of the question. "
    "Never reveal customer identifiers. If something cannot be answered with the tools, say so."
)

FILTERS = {
    "min_prob": {"type": "number", "description": "0 to 1, default 0.8"},
    "max_prob": {"type": "number", "description": "0 to 1, default 1"},
    "channel": {"type": "string", "enum": CHANNELS},
    "merchant_cat": {"type": "string", "enum": MERCHANTS},
    "min_amount": {"type": "number", "description": "RM"},
    "max_amount": {"type": "number", "description": "RM"},
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_kpis",
        "description": "Overall figures: total transactions, fraud transactions, fraud rate, open alerts (probability >= 0.8), fraud by channel.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "alert_stats",
        "description": "Summary of the alerts: how many alerts in total, how many are Priority 1 and how many Priority 2, total and average amount (RM) "
                       "and average probability. Optionally filtered like list_alerts. Defaults to open alerts (>= 0.8).",
        "parameters": {"type": "object", "properties": FILTERS}}},
    {"type": "function", "function": {
        "name": "alerts_breakdown",
        "description": "Alerts grouped by channel, merchant_cat, hour_of_day or priority, with the count, total amount (RM) and "
                       "average probability of each group, largest group first. Defaults to open alerts (>= 0.8).",
        "parameters": {"type": "object", "properties": {
            "group_by": {"type": "string", "enum": list(GROUPS)}, **FILTERS}, "required": ["group_by"]}}},
    {"type": "function", "function": {
        "name": "list_alerts",
        "description": "List individual alerts (max 20), optionally filtered, ordered by probability or amount. "
                       "Defaults to open alerts (>= 0.8), highest probability first.",
        "parameters": {"type": "object", "properties": {
            "order_by": {"type": "string", "enum": ["probability", "amount"]}, "limit": {"type": "integer"}, **FILTERS}}}},
    {"type": "function", "function": {
        "name": "get_alert",
        "description": "Details, probability and reason codes of one transaction by id.",
        "parameters": {"type": "object", "properties": {"txn_id": {"type": "integer"}}, "required": ["txn_id"]}}},
    {"type": "function", "function": {
        "name": "get_model_card",
        "description": "How the fraud model was trained in Teradata: algorithm, AUC, Gini, row counts, training time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_policies",
        "description": "Search the uploaded fraud policies/SOPs (in-database vector search). Returns excerpts with document and section/page.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def _where(args: dict) -> str | dict:
    """WHERE clause for alert filters; only validated numbers and enum values ever reach the SQL text."""
    lo = min(max(float(args.get("min_prob", db.ALERT_THRESHOLD)), 0.0), 1.0)
    hi = min(max(float(args.get("max_prob", 1.0)), 0.0), 1.0)
    parts = [f"s.Prob_1 >= {lo}", f"s.Prob_1 <= {hi}"]
    for key, allowed, column in (("channel", CHANNELS, "t.channel"), ("merchant_cat", MERCHANTS, "t.merchant_cat")):
        if args.get(key) is not None:
            if args[key] not in allowed:
                return {"error": f"unknown {key}; use one of {allowed}"}
            parts.append(f"{column} = '{args[key]}'")
    if args.get("min_amount") is not None:
        parts.append(f"t.amount_myr >= {float(args['min_amount'])}")
    if args.get("max_amount") is not None:
        parts.append(f"t.amount_myr <= {float(args['max_amount'])}")
    return " AND ".join(parts)


FROM = "FROM txn t JOIN txn_scores s ON t.txn_id = s.txn_id"


def _alert_stats(args: dict) -> dict:
    where = _where(args)
    if isinstance(where, dict):
        return where
    sql = ("SELECT COUNT(*) AS n, SUM(CASE WHEN s.Prob_1 >= 0.9 THEN 1 ELSE 0 END) AS p1, "
           "SUM(CASE WHEN s.Prob_1 >= 0.8 AND s.Prob_1 < 0.9 THEN 1 ELSE 0 END) AS p2, "
           f"SUM(t.amount_myr) AS total, AVG(t.amount_myr) AS avg_amt, AVG(s.Prob_1) AS avg_prob {FROM} WHERE {where}")
    r = db.cached(("stats", sql), lambda: db.query_df(sql).iloc[0])
    n = int(r["n"])
    return {"alerts": n, "priority_1": int(r["p1"] or 0), "priority_2": int(r["p2"] or 0),
            "total_amount_myr": round(float(r["total"] or 0), 2), "average_amount_myr": round(float(r["avg_amt"] or 0), 2),
            "average_probability": round(float(r["avg_prob"] or 0), 4)}


def _alerts_breakdown(args: dict) -> dict:
    group = args.get("group_by")
    if group not in GROUPS:
        return {"error": f"group_by must be one of {list(GROUPS)}"}
    where = _where(args)
    if isinstance(where, dict):
        return where
    key = GROUPS[group]
    sql = (f"SELECT {key} AS grp, COUNT(*) AS n, SUM(t.amount_myr) AS total, AVG(s.Prob_1) AS avg_prob "
           f"{FROM} WHERE {where} GROUP BY 1 ORDER BY 2 DESC")
    df = db.cached(("breakdown", sql), lambda: db.query_df(sql))
    return {"group_by": group, "groups": [
        {group: str(r.grp), "alerts": int(r.n), "total_amount_myr": round(float(r.total), 2),
         "average_probability": round(float(r.avg_prob), 4)} for r in df.itertuples(index=False)]}


def _list_alerts(args: dict) -> dict:
    where = _where(args)
    if isinstance(where, dict):
        return where
    limit = min(max(int(args.get("limit", 10)), 1), 20)
    order = "t.amount_myr" if args.get("order_by") == "amount" else "s.Prob_1"
    df = db.query_df(
        f"SELECT TOP {limit} t.txn_id, t.amount_myr, t.channel, t.merchant_cat, s.Prob_1 AS prob "
        f"{FROM} WHERE {where} ORDER BY {order} DESC")
    return {"alerts": [{"txn_id": int(r.txn_id), "amount_myr": float(r.amount_myr), "channel": r.channel,
                        "merchant_cat": r.merchant_cat, "probability": round(float(r.prob), 4)}
                       for r in df.itertuples(index=False)]}


def _get_alert(args: dict) -> dict:
    found = data.get_txn(int(args["txn_id"]))
    if found is None:
        return {"error": f"transaction {args['txn_id']} not found"}
    txn, prob, _ = found
    txn = {k: v for k, v in txn.items() if k != "customer_id"}  # no customer identifiers to the model
    return {"transaction": txn, "probability": round(prob, 4), "reasons": reasons.for_txn(txn)}


def _kpis(_: dict) -> dict:
    from ..routers import kpis
    k = kpis.kpis()
    return {kk: v for kk, v in k.items() if kk != "sql"}


def _model(_: dict) -> dict:
    from ..routers import ml
    m = ml.model_card()
    return {kk: v for kk, v in m.items() if kk != "sql"}


def run_tool(name: str, args: dict, found: list[dict]) -> dict:
    if name == "search_policies":
        hits = rag.retrieve(str(args.get("query", ""))[:500], k=3)
        found.extend(hits)
        return {"excerpts": [{"source": f"{h['doc']} {h['section']}", "text": h["text"]} for h in hits]}
    tools = {"get_kpis": _kpis, "alert_stats": _alert_stats,
             "alerts_breakdown": _alerts_breakdown, "list_alerts": _list_alerts,
             "get_alert": _get_alert, "get_model_card": _model}
    if name not in tools:
        return {"error": f"unknown tool {name}"}
    try:
        return tools[name](args)
    except (KeyError, ValueError, TypeError) as e:
        return {"error": f"bad arguments: {e}"}


def resolve_alert_id(alert_id: int | None, context: str | None) -> int | None:
    """The alert the user is asking about: the explicit id, else a '#12345' in the highlighted text if it exists.
    Raises LookupError when an explicit id does not exist."""
    if alert_id is not None:
        if data.get_txn(alert_id) is None:
            raise LookupError(f"transaction {alert_id} not found")
        return alert_id
    for match in re.finditer(r"#\s?(\d{1,9})\b", context or ""):
        if data.get_txn(int(match.group(1))) is not None:
            return int(match.group(1))
    return None


def focus_block(alert_id: int) -> str:
    """Everything known about one alert, handed to the model as the FOCUSED ALERT (no customer identifier)."""
    txn, prob, _ = data.get_txn(alert_id)
    customer, _ = data.get_customer(txn)
    txn = {k: v for k, v in txn.items() if k != "customer_id"}
    customer = {k: v for k, v in customer.items() if k != "customer_id"}
    return ("FOCUSED ALERT (the user pointed at this alert): " + json.dumps({
        "txn_id": alert_id, "probability": round(prob, 4), "priority": "P1" if prob >= 0.9 else "P2" if prob >= 0.8 else None,
        "transaction": txn, "reason_codes": reasons.for_txn(txn), "customer_summary": customer}, default=str))


def answer(history: list[dict], context: str | None, alert_id: int | None = None) -> dict:
    turns = [dict(m) for m in history]
    if context:
        turns[-1]["content"] += f"\n\nSelected text on screen: \"\"\"{context}\"\"\""
    focus_id = resolve_alert_id(alert_id, context)
    system = SYSTEM + ("\n\n" + focus_block(focus_id) if focus_id is not None else "")
    messages: list[dict] = [{"role": "system", "content": system}, *turns]
    used: list[str] = []
    found: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        msg = llm.chat(messages, TOOLS)
        if not getattr(msg, "tool_calls", None):
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            used.append(call.function.name)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(run_tool(call.function.name, args, found), default=str)})
    else:
        msg = llm.chat(messages)  # out of tool rounds: ask for a final answer without tools
    seen, citations = set(), []
    for h in found:
        if h["chunk_id"] not in seen:
            seen.add(h["chunk_id"])
            citations.append({k: h[k] for k in ("doc", "chunk_id", "section", "page", "text")})
    return {"answer": (msg.content or "").strip(), "citations": citations, "tools": list(dict.fromkeys(used))}
