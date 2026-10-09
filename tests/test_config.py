"""One model setting drives every LLM call; everything is configurable from .env."""
from pathlib import Path

from service.config import Settings
from service.services import llm

ROOT = Path(__file__).resolve().parent.parent
BASE = dict(td_host="h", td_user="u", td_password="p")


def test_one_model_drives_every_llm_call_by_default():
    s = Settings(**BASE, llm_model="model-x")
    assert s.llm_model == s.verify_model == s.gate_model == "model-x"


def test_the_verifier_and_the_topic_gate_can_still_be_overridden_when_wanted():
    s = Settings(**BASE, llm_model="model-x", verify_model="strong", gate_model="fast")
    assert (s.llm_model, s.verify_model, s.gate_model) == ("model-x", "strong", "fast")


def test_blank_overrides_in_the_env_file_mean_use_the_main_model():
    s = Settings(**BASE, llm_model="model-x", verify_model="", gate_model="")
    assert s.verify_model == s.gate_model == "model-x"


def test_env_example_documents_every_setting_you_may_change():
    keys = {line.split("=")[0].strip() for line in (ROOT / ".env.example").read_text().splitlines() if "=" in line and not line.startswith("#")}
    assert {"TD_HOST", "TD_USER", "TD_PASSWORD", "OPENAI_API_KEY", "LLM_MODEL", "REASONING_EFFORT", "EMBED_MODEL"} <= keys
    assert ".env.example" not in (ROOT / ".gitignore").read_text()  # the example is committed, the real .env is not


def test_changing_llm_model_in_the_environment_changes_the_model_every_call_uses(monkeypatch):
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw["model"])
            from types import SimpleNamespace
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.delenv("VERIFY_MODEL", raising=False)
    monkeypatch.delenv("GATE_MODEL", raising=False)
    llm.complete("s", "u")           # the quick topic check
    llm.complete_json("s", "u")      # brief answers
    llm.complete_json("s", "u", llm.get_settings().verify_model)  # the relevance check
    llm.chat([{"role": "user", "content": "x"}])  # chat
    assert seen == ["env-model"] * 4


def test_gpt5_models_get_reasoning_options_and_others_get_temperature_zero(monkeypatch):
    monkeypatch.setenv("REASONING_EFFORT", "medium")
    five = llm._options("gpt-5-mini")
    assert five["reasoning_effort"] == "medium" and "temperature" not in five
    older = llm._options("gpt-4o-mini")
    assert older["temperature"] == 0 and "reasoning_effort" not in older
