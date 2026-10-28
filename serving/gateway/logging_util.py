from __future__ import annotations

import json
import logging
import sys
from typing import Any

_logger = logging.getLogger("serving.gateway")
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    _logger.info(json.dumps(payload, default=str, separators=(",", ":")))
