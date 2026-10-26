"""Which tools the agent gets. One place, so the terminal and the API stay in step."""

from __future__ import annotations

from bloom.search_tool import semantic_search
from bloom.tools import TOOLS

# Every tool the agent can use: the file tools plus search by meaning.
AGENT_TOOLS = [*TOOLS, semantic_search]
