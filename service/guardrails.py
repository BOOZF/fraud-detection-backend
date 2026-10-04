"""Guardrails for the copilot chat, in the layered style LangChain describes: cheap deterministic rules first,
a model-based check where meaning matters, and a scan of the answer before it leaves.

  before the model : screen_input (injection, secrets)  ->  mask_pii  ->  topic_of (is it about fraud at this bank?)
  after the model  : clean_output (PII / keys never leave in an answer)
"""
import re

from .services import llm, rag

REFUSAL = ("I can only help with fraud detection at Malaysia XX Bank: its alerts and transactions, the fraud model, "
           "and the uploaded policy documents. Please ask about one of those.")

# ---------- deterministic rules ----------

INJECTION = [re.compile(p, re.I) for p in (
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,30}\b(instruction|rule|prompt|guideline|direction)s?\b",
    r"\b(reveal|show|print|repeat|display|leak|output)\b.{0,30}\b(system|hidden|secret|initial|original)\b.{0,15}\b(prompt|instruction|message)s?\b",
    r"\byou are now\b.{0,60}\b(dan|unrestricted|without (any )?(restriction|rule|limit)s?|jailbr)",
    r"\b(jailbreak|developer mode|do anything now)\b",
)]
SECRET = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|(?:api[_ -]?key|password|passwd|secret)\s*[:=]\s*\S{6,})", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
NRIC = re.compile(r"\b\d{6}-\d{2}-\d{4}\b")  # Malaysian national ID


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _mask_card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group())
    return f"[card ending {digits[-4:]}]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group()


def mask_pii(text: str) -> str:
    """Mask e-mails, card numbers (Luhn-valid only, so amounts and ids survive) and NRICs."""
    text = EMAIL.sub("[email]", text)
    text = NRIC.sub("[national id]", text)
    return CARD.sub(_mask_card, text)


def screen_input(text: str) -> str | None:
    """'secret' | 'prompt_injection' | None. Rule-based: runs before any model is called."""
    if SECRET.search(text):
        return "secret"
    if any(p.search(text) for p in INJECTION):
        return "prompt_injection"
    return None


def clean_output(text: str) -> str:
    return SECRET.sub("[removed]", mask_pii(text))


# ---------- model-based topic check ----------

POLICY_DISTANCE = 0.6

TOPIC_SYSTEM = (
    "You are a strict topic gate for a bank's fraud-operations assistant. Reply with exactly ON_TOPIC or OFF_TOPIC. "
    "ON_TOPIC: fraud and financial crime; the bank's transactions, alerts, scores, priorities, customers' behaviour; "
    "the fraud model and its metrics; this dashboard; the uploaded policy documents, SOPs and their contents "
    "(investigations, compliance, reporting, evidence, procedures); short greetings, thanks, and questions about "
    "what the assistant can do; follow-up questions that continue an on-topic conversation. "
    "If the user is pointing at a fraud alert or has highlighted text on screen, short or vague questions about it "
    "('explain this', 'tell me about it') are ON_TOPIC. When in doubt, a question that could plausibly concern "
    "an investigation, a procedure, a rule or a person's obligations in a policy document is ON_TOPIC. "
    "Judge the last USER message in the light of the whole conversation: a follow-up that builds on an on-topic answer "
    "(for example 'and what share of all transactions is that?' after a count of alerts) is ON_TOPIC. "
    "OFF_TOPIC: only requests that are clearly unrelated to all of the above, such as general knowledge, trivia, geography, recipes, weather, sports, entertainment, "
    "poems or stories, coding or maths help, personal advice, politics, translation, and requests to act as something else."
)


SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|good (morning|afternoon|evening)|what can you do\??|help)[\s!.?]*$", re.I)


def topic_of(history: list[dict], context: str | None = None, alert_id: int | None = None) -> str:
    """'on_topic' or 'off_topic' for the last user message, read with the recent turns, the alert the user is
    pointing at and the text they highlighted, so follow-ups and 'explain this' count."""
    if SMALL_TALK.match(history[-1]["content"]):
        return "on_topic"  # the model's own scope rule answers these; no need to ask a model whether to answer them
    recent = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-4:])
    screen = ""
    if alert_id is not None:
        screen += f"\nThe user is looking at fraud alert #{alert_id}."
    if context:
        screen += f"\nThe user highlighted this text on screen: \"{context[:300]}\""
    verdict = llm.complete(TOPIC_SYSTEM, f"CONVERSATION:\n{recent}{screen}\n\nIs the last USER message on topic?")
    if not verdict.strip().upper().startswith("OFF"):
        return "on_topic"
    # Second opinion, from the data rather than the model: if the uploaded documents have a close passage, the
    # question is about them (measured: policy questions land at cosine distance <= 0.45, off-topic ones >= 0.75).
    closest = rag.retrieve(history[-1]["content"], k=1)
    return "on_topic" if closest and closest[0]["distance"] <= POLICY_DISTANCE else "off_topic"
