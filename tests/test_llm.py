from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from bloom.llm import ask_model, build_chat_model, format_usage
from bloom.prompts import SYSTEM_PROMPT


class StubModel:
    """Stands in for a chat model: records what it is sent and returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.received = None

    def invoke(self, messages):
        self.received = messages
        return self.reply


def test_ask_model_sends_system_prompt_then_question():
    stub = StubModel(AIMessage(content="pong"))
    reply = ask_model(stub, "ping")
    assert reply.text == "pong"
    assert [type(m) for m in stub.received] == [SystemMessage, HumanMessage]
    assert stub.received[0].content == SYSTEM_PROMPT
    assert stub.received[1].content == "ping"


def test_build_chat_model_uses_the_override(monkeypatch):
    monkeypatch.setattr("bloom.llm.settings.gemini_api_key", SecretStr("fake-key"))
    model = build_chat_model("my-test-model")
    assert model.model.endswith("my-test-model")


def test_format_usage_with_numbers():
    reply = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
    )
    assert format_usage(reply) == "tokens: 12 in, 5 out, 17 total"


def test_format_usage_when_missing():
    assert format_usage(AIMessage(content="x")) == "tokens: not reported"
