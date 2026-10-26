"""Experiment: a model sees only the messages you send it in that one call."""

from langchain_core.messages import HumanMessage

from bloom.llm import build_chat_model

model = build_chat_model()

history = [HumanMessage(content="My name is Alex. Just reply OK.")]
reply = model.invoke(history)
print(f"1 (sent {len(history)} message):", reply.text)

alone = [HumanMessage(content="What is my name?")]
print(f"2 (sent {len(alone)} message):", model.invoke(alone).text)

history.append(reply)
history.append(HumanMessage(content="What is my name?"))
print(f"3 (sent {len(history)} messages):", model.invoke(history).text)
