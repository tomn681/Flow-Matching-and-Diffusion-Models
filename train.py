"""
Compatibility wrapper so `python train.py ...` works from the repo root.
Delegates to the package entrypoint at `src.train`.
"""

from src.train import main


if __name__ == "__main__":
    main()
