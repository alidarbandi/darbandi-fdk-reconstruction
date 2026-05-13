from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "microct_tigre_gui"
sys.path.insert(0, str(PROJECT_DIR))

from main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
