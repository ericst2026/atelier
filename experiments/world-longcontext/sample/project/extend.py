"""Your extension."""


def extend(model, target_length: int) -> None:
    """Modify the model in place so it works at target_length."""
    factor = max(1.0, target_length / model.config.block_size)
    model.config.rope_scale = factor      # position interpolation
    model.config.block_size = target_length
    model._cos = model._sin = None        # drop the cached angles so they are rebuilt
