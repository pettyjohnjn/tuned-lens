"""Backward-compat facade for the old train_loop.py module."""
from __future__ import annotations

# Re-export the public API so `from train_loop import Train` still works.
from .trainer import Train, State, LossChoice, LensVariant  # lightweight
__all__ = ["Train", "State", "LossChoice", "LensVariant"]

def main() -> None:
    """Preserve CLI behavior if users executed this file directly."""
    # Localize heavy deps to runtime only.
    from simple_parsing import ArgumentParser
    import tuned_lens.scripts.ingredients as ing

    import logging
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_arguments(ing.Model, dest="model")
    parser.add_arguments(ing.Data, dest="data")
    parser.add_arguments(ing.Optimizer, dest="opt")
    parser.add_arguments(ing.Distributed, dest="dist")
    parser.add_arguments(Train, dest="train")

    args = parser.parse_args()
    train: Train = args.train
    train.execute()

if __name__ == "__main__":
    main()