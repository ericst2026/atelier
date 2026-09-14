"""Post-training quantisation, written out rather than called.

Weights are stored as int8 or int4 with a scale per output channel and cast back
to bf16 on the fly. Nothing here is fast — the point is to see exactly what is
thrown away and what it costs in loss."""
import torch
import torch.nn as nn


def quantize_tensor(w: torch.Tensor, bits: int = 8, per_channel: bool = True, symmetric: bool = True):
    """Returns (codes, scale, zero) such that w ≈ (codes - zero) * scale."""
    qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
    dim = 1 if per_channel and w.dim() == 2 else None
    if symmetric:
        amax = w.abs().amax(dim=dim, keepdim=True) if dim is not None else w.abs().amax()
        scale = (amax / qmax).clamp(min=1e-8)
        codes = torch.round(w / scale).clamp(-qmax - 1, qmax)
        zero = torch.zeros_like(scale)
    else:
        lo = w.amin(dim=dim, keepdim=True) if dim is not None else w.amin()
        hi = w.amax(dim=dim, keepdim=True) if dim is not None else w.amax()
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zero = torch.round(-lo / scale)
        codes = torch.round(w / scale + zero).clamp(0, qmax)
    return codes.to(torch.int8 if bits <= 8 else torch.int16), scale, zero


def dequantize(codes, scale, zero, dtype=torch.float32):
    return ((codes.to(dtype) - zero.to(dtype)) * scale.to(dtype))


class QuantLinear(nn.Module):
    """A Linear whose weights live as integers and are expanded at each forward pass."""

    def __init__(self, linear: nn.Linear, bits: int = 8, per_channel: bool = True):
        super().__init__()
        codes, scale, zero = quantize_tensor(linear.weight.data.float(), bits, per_channel)
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.bias = linear.bias
        self.bits = bits
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        w = dequantize(self.codes, self.scale, self.zero, x.dtype)
        return nn.functional.linear(x, w, self.bias)

    def error(self, original: torch.Tensor) -> dict:
        w = dequantize(self.codes, self.scale, self.zero, torch.float32)
        diff = (w - original.float())
        return {"mse": float((diff**2).mean()), "max_abs": float(diff.abs().max()), "rel": float(diff.norm() / original.float().norm())}


def quantize_model(model, bits: int = 8, per_channel: bool = True, skip: tuple[str, ...] = ("head",)) -> dict:
    """Replace every Linear except those named in `skip`. Returns per-layer error."""
    report = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                original = child.weight.data.clone()
                q = QuantLinear(child, bits, per_channel).to(child.weight.device)
                setattr(module, child_name, q)
                report[full] = q.error(original)
    return report


def model_bytes(model, bits_for_quant: int = 8) -> dict:
    """Weight footprint, counting quantised layers at their real width."""
    total, quantised = 0, 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            n = m.codes.numel()
            quantised += n
            total += n * bits_for_quant / 8 + m.scale.numel() * 4
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return {"bytes": int(total), "quantised_weights": quantised}
