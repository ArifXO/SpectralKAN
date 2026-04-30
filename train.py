"""Root entry point for MAE pretraining.

This keeps the documented command ``python train.py --config ...`` working
while the implementation lives in ``scripts/train.py``.
"""

from scripts.train import main


if __name__ == "__main__":
    main()
