"""
Training entry aligned with `ehr-ai-system` layout.

Delegates to `training.train` (same CLI). Prefer:

    python -m models.train
    python -m training.train
"""

from training.train import main

if __name__ == "__main__":
    main()
