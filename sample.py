"""
Compatibility wrapper so `python sample.py ...` works from the repo root.
Delegates to the package entrypoint at `src.sample`.
"""

from src.sample import main


if __name__ == "__main__":
    main()
