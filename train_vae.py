"""
Compatibility wrapper so `python train_vae.py` keeps working after moving the
implementation under `src.pipelines`.
"""

from src.pipelines.train.vae import main


if __name__ == "__main__":
    main()
