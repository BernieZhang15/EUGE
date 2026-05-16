from .config import MixLoraConfig
from .utils import is_package_available

assert is_package_available("torch", "2.3.0"), "MixLoRA requires torch>=2.3.0"
assert is_package_available(
    "transformers", "4.42.0"
), "MixLoRA requires transformers>=4.42.0"

__all__ = [
    "MixLoraConfig",
    "MixLoraModelForCausalLM",
    "inject_adapter_in_model",
    "load_adapter_weights",
]


def __getattr__(name):
    if name in {"MixLoraModelForCausalLM", "inject_adapter_in_model", "load_adapter_weights"}:
        from .model import (
            MixLoraModelForCausalLM,
            inject_adapter_in_model,
            load_adapter_weights,
        )

        exports = {
            "MixLoraModelForCausalLM": MixLoraModelForCausalLM,
            "inject_adapter_in_model": inject_adapter_in_model,
            "load_adapter_weights": load_adapter_weights,
        }
        return exports[name]
    raise AttributeError(f"module 'mixlora' has no attribute '{name}'")
