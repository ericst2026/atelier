"""Your parameter-efficient scheme."""
from atelier_mini.lora import apply_lora


def adapt(model) -> dict:
    """Freeze what you like, make trainable what you like, return a census with `trainable`."""
    return apply_lora(model, r=16, alpha=32, targets=("qkv", "proj", "gate", "up", "down"))
