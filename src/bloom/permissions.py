"""The gate between 'the model asked' and 'bloom does it'.

Every tool call is checked before it runs and gets one of three answers:
  ALLOW  run it now          DENY  never run it       ASK  run it only if the human says yes
Deny always wins over ask, and ask always wins over allow.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from bloom.tools import resolve_inside


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Tools that only look and never change anything.
READ_ONLY_TOOLS = {"list_files", "read_file", "find_files", "search_text", "semantic_search"}
# Tools that take a `path` argument.
PATH_TOOLS = {"list_files", "read_file", "search_text", "write_file", "edit_file"}
# Files and folders the model may never touch (the .env holds your API key).
PROTECTED = {".env", ".git"}
# Command fragments that are never allowed, even if the human would say yes.
FORBIDDEN_COMMANDS = ["sudo", "rm -rf", "rm -fr", "mkfs", "shutdown", "reboot", "curl ", "wget "]


def _touches_protected(path: str) -> bool:
    try:
        parts = resolve_inside(path).relative_to(Path.cwd().resolve()).parts
    except ValueError:
        return False  # outside the folder: the tool itself refuses that
    return bool(parts) and parts[0] in PROTECTED


def check(tool_name: str, args: dict) -> tuple[Decision, str]:
    """Decide what to do about one tool call. Returns the decision and a short reason."""
    if tool_name in PATH_TOOLS and _touches_protected(str(args.get("path", "."))):
        return Decision.DENY, f"'{args.get('path')}' is protected"
    if tool_name == "run_command":
        command = str(args.get("command", ""))
        for fragment in FORBIDDEN_COMMANDS:
            if fragment in command:
                return Decision.DENY, f"commands containing '{fragment.strip()}' are never allowed"
    if tool_name in READ_ONLY_TOOLS:
        return Decision.ALLOW, "read-only"
    return Decision.ASK, "this changes things"  # new or unknown tools are asked about by default
