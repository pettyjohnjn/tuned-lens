# hook_points.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import re, os
import torch
from transformers import PreTrainedModel

HookType = str  # "residual_out" | "attn_out" | "mlp_out"

# Regexes that match the *final projection* modules for GPT-2 style stacks.
DEFAULT_PATTERNS: Dict[HookType, List[str]] = {
    "attn_out": [r"transformer\.h\.(\d+)\.attn\.c_proj"],
    "mlp_out":  [r"transformer\.h\.(\d+)\.mlp\.c_proj"],
    # residual_out uses output_hidden_states (no hooks needed)
}


def _find_modules_by_patterns(model: PreTrainedModel, patterns: List[str]) -> List[torch.nn.Module]:
    compiled = [re.compile(p) for p in patterns]
    matches: List[tuple[int, str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        for rx in compiled:
            m = rx.fullmatch(name)
            if m:
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = 10_000_000
                matches.append((idx, name, module))
                break
    if not matches:
        raise RuntimeError(f"No modules matched any of: {patterns}")
    matches.sort(key=lambda t: (t[0], t[1]))
    return [m for _, _, m in matches]


@dataclass
class HookStorage:
    """CPU-offloaded per-layer buffers mirroring hidden-state path."""
    offload_to_cpu: bool
    offload_dtype: torch.dtype
    pin_memory: bool
    buffers: List[torch.Tensor]

    def __init__(self, offload_to_cpu: bool, offload_dtype: torch.dtype, pin_memory: bool):
        self.offload_to_cpu = offload_to_cpu
        self.offload_dtype = offload_dtype
        self.pin_memory = pin_memory
        self.buffers = []

    def append(self, x: torch.Tensor):
        if self.offload_to_cpu:
            x_cpu = x.detach().to("cpu", dtype=self.offload_dtype, non_blocking=True, copy=True)
            if self.pin_memory:
                x_cpu = x_cpu.pin_memory()
            self.buffers.append(x_cpu)
        else:
            self.buffers.append(x.detach())

    def clear(self):
        for t in self.buffers:
            del t
        self.buffers = []


class HookManager:
    """
    Registers forward hooks for 'attn_out' and 'mlp_out' on their *final projection*
    modules. 'residual_out' remains handled via output_hidden_states in the train loop.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        hook_points: List[HookType],
        offload_to_cpu: bool,
        offload_dtype: torch.dtype,
        pin_memory: bool,
        override_patterns: Optional[Dict[HookType, List[str]]] = None,
    ):
        self.model = model
        self.hook_points = hook_points
        self.offload_to_cpu = offload_to_cpu
        self.offload_dtype = offload_dtype
        self.pin_memory = pin_memory
        self.patterns = override_patterns or DEFAULT_PATTERNS

        self.handles: Dict[HookType, List[torch.utils.hooks.RemovableHandle]] = {}
        self.storage: Dict[HookType, HookStorage] = {}

    def _make_cb(self, hook_type: HookType, storage: HookStorage):
        def _cb(_mod, _inp, out):
            if isinstance(out, (tuple, list)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        out = item
                        break
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Hook {hook_type}: expected Tensor output, got {type(out)}")
            storage.append(out)
            return None
        return _cb

    def register(self):
        for hp in self.hook_points:
            if hp not in ("attn_out", "mlp_out"):
                continue
            modules = _find_modules_by_patterns(self.model, self.patterns.get(hp, []))
            storage = HookStorage(self.offload_to_cpu, self.offload_dtype, self.pin_memory)
            self.storage[hp] = storage
            self.handles[hp] = [m.register_forward_hook(self._make_cb(hp, storage)) for m in modules]

    def get_buffers(self, hook_type: HookType) -> List[torch.Tensor]:
        return self.storage[hook_type].buffers

    def clear(self):
        for st in self.storage.values():
            st.clear()

    def remove(self):
        for hs in self.handles.values():
            for h in hs:
                h.remove()
        self.handles.clear()
        self.storage.clear()

    # -------- NEW: save hook activations into per-hook subfolders --------
    def save(self, base_dir: str | os.PathLike, tag: Optional[str] = None):
        """
        Save current in-memory hook activations to disk in subfolders:
            <base_dir>/<hook_type>/layer_{i}[__{tag}].pt
        """
        os.makedirs(base_dir, exist_ok=True)
        suffix = f"__{tag}" if tag else ""
        for hook_name, storage in self.storage.items():
            out_dir = os.path.join(base_dir, hook_name)
            os.makedirs(out_dir, exist_ok=True)
            for i, tensor in enumerate(storage.buffers):
                fname = f"layer_{i}{suffix}.pt"
                torch.save(tensor, os.path.join(out_dir, fname))