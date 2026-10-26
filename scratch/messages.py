from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

conversation = [
    SystemMessage(content="You are bloom, a coding assistant."),
    HumanMessage(content="How many files are in the current folder?"),
    # The model does not run anything. It answers with a *request* to call a tool.
    AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "call_1"}],
    ),
    # Our program runs the tool and sends the result back, tagged with the same id.
    ToolMessage(content="README.md\npyproject.toml\nsrc", tool_call_id="call_1"),
    AIMessage(content="There are 3 items: README.md, pyproject.toml and src."),
]

for message in conversation:
    print(f"{type(message).__name__:14} {message.content!r}")
    for call in getattr(message, "tool_calls", []):
        print(f"{'':14} -> requests tool {call['name']} with {call['args']}")
