from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryLog:
    def __init__(self, callback: Callable[[str], None] | None = None) -> None:
        self.lines: list[str] = []
        self.callback = callback

    def write(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        self.lines.append(line)
        if self.callback is not None:
            self.callback(line)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.write(message)

    def save(self, path: str | Path) -> Path:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return log_path


def write_processing_log(path: str | Path, lines: Iterable[str]) -> Path:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path
