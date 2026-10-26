"""See context shrinking happen, with no model and no network."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bloom.context import estimate_tokens, shrink_old_tool_results

messages = [HumanMessage(content="Read six files for me.")]
for i in range(6):
    call = {"name": "read_file", "args": {"path": f"file{i}.py"}, "id": f"call_{i}"}
    messages.append(AIMessage(content="", tool_calls=[call]))
    messages.append(ToolMessage(content="x = 1\n" * 2000, tool_call_id=f"call_{i}"))

smaller = shrink_old_tool_results(messages)

print("before:", estimate_tokens(messages), "tokens")
print("after: ", estimate_tokens(smaller), "tokens")
for message in smaller:
    if isinstance(message, ToolMessage):
        print(message.tool_call_id, "->", message.content[:60].replace("\n", " "))
