"""Pluggable LLM adapter. Only transaction facts, score, reason codes and retrieved policy text
are sent; swapping to an in-country model is a config change (OpenAI-compatible base URL)."""
import json
from types import SimpleNamespace

from openai import OpenAI

from ..config import get_settings

SEED = 20261004  # fixed so the same input gives the same wording as far as the model allows


def _options(model: str, effort: str | None = None) -> dict:
    """gpt-5 models are reasoning models: no temperature, a reasoning effort and a token budget that includes the
    reasoning. Older models are run at temperature 0."""
    if model.startswith("gpt-5"):
        return {"reasoning_effort": effort or get_settings().reasoning_effort, "max_completion_tokens": 8000, "seed": SEED}
    return {"temperature": 0, "seed": SEED, "max_tokens": 3000}  # a cap: older models can ramble on until they run out


def complete(system: str, user: str, model: str | None = None) -> str:
    """A short plain-text completion (the quick topic check): the configured model, with the least reasoning."""
    s = get_settings()
    name = model or s.gate_model
    client = OpenAI(api_key=s.openai_api_key, timeout=60)
    r = client.chat.completions.create(
        model=name, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **_options(name, effort="minimal"))
    return (r.choices[0].message.content or "").strip()


def complete_json(system: str, user: str, model: str | None = None) -> dict:
    s = get_settings()
    name = model or s.llm_model
    client = OpenAI(api_key=s.openai_api_key, timeout=180)
    r = client.chat.completions.create(
        model=name, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **_options(name))
    return json.loads(r.choices[0].message.content)


def chat(messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False):
    """One chat-completions turn, optionally with function tools. With json_mode the final answer must be a JSON
    object (tool calls are unaffected). Returns the assistant message."""
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=180)
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(model=s.llm_model, messages=messages, **_options(s.llm_model), **kwargs).choices[0].message


def chat_stream(messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False):
    """Like chat(), but streamed: yields ("delta", text) as the answer is written, then ("message", message) with the
    whole assistant message (content and any tool calls) assembled from the stream."""
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=180)
    kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    stream = client.chat.completions.create(model=s.llm_model, messages=messages, stream=True, **_options(s.llm_model), **kwargs)
    parts: list[str] = []
    calls: dict[int, dict] = {}
    for part in stream:
        if not part.choices:
            continue
        delta = part.choices[0].delta
        if delta.content:
            parts.append(delta.content)
            yield ("delta", delta.content)
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            call["id"] = tc.id or call["id"]
            if tc.function:
                call["name"] += tc.function.name or ""
                call["arguments"] += tc.function.arguments or ""
    tool_calls = [SimpleNamespace(id=c["id"], function=SimpleNamespace(name=c["name"], arguments=c["arguments"]))
                  for _, c in sorted(calls.items())]
    yield ("message", SimpleNamespace(content="".join(parts) or None, tool_calls=tool_calls or None))
