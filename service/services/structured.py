"""The one shape every chat answer has. The model fills a small JSON schema; this module renders it to Markdown in a
fixed order, so the layout never depends on how the model felt that day.

    {"kind": "alert" | "data" | "policy" | "general",
     "summary": "one sentence", "sections": [{"heading": "...", "points": ["..."]}],
     "table": {"headers": [...], "rows": [[...]]} | null, "takeaway": "one sentence" | null}
"""
import json
import re

# The only section headings each kind of answer may have, in the order they are shown.
HEADINGS = {
    "alert": ["Key facts", "Why it was flagged", "Policy guidance", "Suggested next steps"],
    "data": ["Notes"],
    "policy": ["What the policy says", "How it applies"],
    "general": [],
}
MAX_SUMMARY_WORDS, MAX_POINTS, MAX_POINT_WORDS, MAX_TAKEAWAY_WORDS = 40, 5, 35, 30
MAX_TABLE_ROWS, MAX_TABLE_COLUMNS = 21, 6
NUMERIC = re.compile(r"^(RM\s?)?[-+]?[\d,]*\.?\d+\s?%?$")


def _words(text, limit: int) -> str:
    words = " ".join(str(text).split()).split(" ")
    return " ".join(words[:limit]) + ("..." if len(words) > limit else "")


def parse(text: str) -> dict | None:
    """The model's reply as a dict, or None if it is not a usable answer (not JSON, or no summary)."""
    body = (text or "").strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    try:
        reply = json.loads(body)
    except ValueError:
        return None
    if not isinstance(reply, dict) or not str(reply.get("summary", "")).strip():
        return None
    return reply


def _cell(value) -> str:
    return " ".join(str(value).replace("|", "/").split())


def _table(table) -> str | None:
    if not isinstance(table, dict) or not table.get("headers") or not isinstance(table.get("rows"), list):
        return None
    headers = [_cell(h) for h in table["headers"]][:MAX_TABLE_COLUMNS]
    rows = [[_cell(c) for c in list(r)[:len(headers)]] for r in table["rows"][:MAX_TABLE_ROWS] if isinstance(r, (list, tuple))]
    rows = [r + [""] * (len(headers) - len(r)) for r in rows]
    numeric = [bool(rows) and all(NUMERIC.match(r[i]) for r in rows) for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(reply: dict) -> str:
    kind = reply.get("kind") if reply.get("kind") in HEADINGS else "general"
    blocks = [_words(reply.get("summary") or "No details available.", MAX_SUMMARY_WORDS)]
    if kind == "general":
        return blocks[0]
    table = _table(reply.get("table"))
    if table:
        blocks.append(table)
    by_heading = {}
    for section in reply.get("sections") or []:
        if isinstance(section, dict):
            by_heading.setdefault(" ".join(str(section.get("heading", "")).split()).lower(), section)
    for heading in HEADINGS[kind]:
        points = [_words(p, MAX_POINT_WORDS) for p in (by_heading.get(heading.lower(), {}).get("points") or []) if str(p).strip()]
        if points:
            blocks.append(f"**{heading}**\n" + "\n".join(f"- {p}" for p in points[:MAX_POINTS]))
    if reply.get("takeaway"):
        blocks.append(_words(reply["takeaway"], MAX_TAKEAWAY_WORDS))
    return "\n\n".join(blocks)


# ---------- partial JSON, for streaming ----------

def _complete(text: str) -> str | None:
    """Best-effort completion of a JSON object that is still being written: close the open string and brackets; if the
    cut falls somewhere that cannot be closed (after a key, say), fall back to the last comma before it."""
    stack, cuts, in_str, esc = [], [], False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    head = text
    if in_str:
        head = re.sub(r"\\u[0-9a-fA-F]{0,3}$", "", head)  # a unicode escape cut in half
        if esc:
            head = head[:-1]  # a lone backslash
        head += '"'
    candidates = [head + "".join(reversed(stack))]
    candidates += [text[:i] + closers for i, closers in reversed(cuts[-8:])]
    for candidate in candidates:
        try:
            if isinstance(json.loads(candidate), dict):
                return candidate
        except ValueError:
            continue
    return None


def parse_prefix(text: str) -> dict | None:
    """The reply so far as a dict, or None while there is no summary to show yet."""
    body = re.sub(r"^```(?:json)?\s*", "", (text or "").strip())
    completed = _complete(body)
    if completed is None:
        return None
    reply = json.loads(completed)
    return reply if str(reply.get("summary") or "").strip() else None


def render_partial(text: str) -> str | None:
    reply = parse_prefix(text)
    return render(reply) if reply else None
