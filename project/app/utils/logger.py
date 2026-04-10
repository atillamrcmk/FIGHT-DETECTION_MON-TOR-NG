from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class JsonEventLogger:
    def __init__(self, logs_dir: str) -> None:
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

    def log_event(self, event: Dict[str, Any], filename_prefix: str = "events") -> str:
        date = time.strftime("%Y%m%d", time.localtime())
        path = os.path.join(self.logs_dir, f"{filename_prefix}_{date}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return path


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return str(obj)

