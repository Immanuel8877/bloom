"""The agent loop, drawn as a LangGraph graph.

START -> model --(tool calls?)--> tools --> model --> ... --> END
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from bloom.context import prepare_for_model
from bloom.permissions import Decision, check
from bloom.prompts import SYSTEM_PROMPT

MAX_STEPS = 25

# Asked before an action that changes things: (tool name, arguments) -> did the human say yes?
Approver = Callable[[str, dict], bool]


def build_graph(
    chat_model: BaseChatModel, tools: Sequence[BaseTool], approve: Approver | None = None
):
    """Wire the model and the tools into a graph and return it, ready to run."""
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = chat_model.bind_tools(list(tools))

    def call_model(state: MessagesState) -> dict:
        """Node 1: show the model the conversation so far and add its reply."""
        reply = model_with_tools.invoke(prepare_for_model(state["messages"]))
        return {"messages": [reply]}

    def run_tools(state: MessagesState) -> dict:
        """Node 2: run every tool the last reply asked for, one ToolMessage each."""
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            decision, reason = check(call["name"], call["args"])
            if tool is None:
                text = f"Error: there is no tool called '{call['name']}'"
            elif decision is Decision.DENY:
                text = f"Denied: {reason}. Do not try again; tell the user instead."
            elif decision is Decision.ASK and not (approve and approve(call["name"], call["args"])):
                text = "Declined: the user did not approve this action."
            else:
                try:
                    text = str(tool.invoke(call["args"]))
                except Exception as exc:  # a broken tool must not crash the loop
                    text = f"Error: {type(exc).__name__}: {exc}"
            results.append(ToolMessage(content=text, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def should_continue(state: MessagesState) -> str:
        """The edge: did the model ask for tools? Then run them. Otherwise we are done."""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_continue, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def new_conversation() -> list[BaseMessage]:
    """A fresh conversation: just the system prompt."""
    return [SystemMessage(content=SYSTEM_PROMPT)]


def run_turn(graph, history: list[BaseMessage], text: str) -> list[BaseMessage]:
    """Add the user's `text` to `history`, run the loop, and return the longer history."""
    messages = [*history, HumanMessage(content=text)]
    result = graph.invoke({"messages": messages}, {"recursion_limit": MAX_STEPS})
    return result["messages"]


def run_agent(
    chat_model: BaseChatModel,
    tools: Sequence[BaseTool],
    question: str,
    approve: Approver | None = None,
) -> list[BaseMessage]:
    """Run the loop for one question and return the whole conversation."""
    graph = build_graph(chat_model, tools, approve)
    return run_turn(graph, new_conversation(), question)


def final_answer(messages: list[BaseMessage]) -> str:
    """The text of the last AI message."""
    last = messages[-1]
    return last.text if isinstance(last, AIMessage) else ""
