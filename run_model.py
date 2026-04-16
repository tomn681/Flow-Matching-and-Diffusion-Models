"""
Compatibility wrapper so `python run_model.py ...` works from the repo root.
Delegates to the package entrypoint at `src.run_model`.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.run_model import main


if __name__ == "__main__":
    main()
