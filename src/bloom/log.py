from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_FILE = Path(".bloom") / "logs" / "bloom.log"

_configured = False


def init_logger(level: str = "INFO", log_file: Path = DEFAULT_LOG_FILE) -> None:

    global _configured
    if _configured:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    _configured = True
