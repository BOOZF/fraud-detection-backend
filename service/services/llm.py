"""Pluggable LLM adapter. Only transaction facts, score, reason codes and retrieved policy text
are sent; swapping to an in-country model is a config change (OpenAI-compatible base URL)."""
import json

from openai import OpenAI

from ..config import get_settings


def complete(system: str, user: str) -> str:
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=30)
    r = client.chat.completions.create(
        model=s.llm_model, temperature=0.1,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content.strip()


def complete_json(system: str, user: str) -> dict:
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=90)
    r = client.chat.completions.create(
        model=s.llm_model, temperature=0.1, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)


def chat(messages: list[dict], tools: list[dict] | None = None):
    """One chat-completions turn, optionally with function tools. Returns the assistant message."""
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=90)
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    return client.chat.completions.create(model=s.llm_model, temperature=0.1, messages=messages, **kwargs).choices[0].message
