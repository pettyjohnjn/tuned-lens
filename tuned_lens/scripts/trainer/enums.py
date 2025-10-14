import enum

class LossChoice(enum.Enum):
    """Options of what loss to select when training the model."""
    CE = "ce"
    KL = "kl"
    SUBSET_KL = "subset_kl"

class LensVariant(enum.Enum):
    TUNED = "tuned"
    LORA = "lora"

__all__ = ["LossChoice", "LensVariant"]