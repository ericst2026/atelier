"""Your inference stack."""
from atelier_mini.gen import generate_cached
from atelier_mini.quant import quantize_model


class Optimised:
    def __init__(self, model, tok, system=None):
        self.model, self.tok, self.system = model, tok, system

    def generate(self, prompts, max_new_tokens=160):
        return generate_cached(self.model, self.tok, prompts, max_new_tokens, 0.0, system=self.system)[0]


def optimise(model, tok, system=None):
    quantize_model(model, bits=8, per_channel=True, skip=("head",))
    return Optimised(model, tok, system)
