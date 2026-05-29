import importlib.metadata
import importlib.util
import json
import logging
import os
from typing import Dict, List, Optional

import torch
from packaging import version


def get_mixlora_moe_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    cached = getattr(model, "_mixlora_moe_modules", None)
    if cached is not None:
        return cached

    modules = [
        module
        for module in model.modules()
        if getattr(module, "_is_mixlora_moe", False)
    ]
    setattr(model, "_mixlora_moe_modules", modules)
    return modules


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_device_spec(device) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, str):
        normalized = device.strip()
        if not normalized:
            raise ValueError("device must be a non-empty string")
        return normalized
    raise ValueError("device must be a string")


def resolve_device(cfg: Optional[dict] = None) -> str:
    cfg = cfg or {}
    config_device = normalize_device_spec(cfg.get("device"))
    if config_device is not None:
        return config_device

    return infer_device()


def infer_dtype(device: Optional[str] = None) -> torch.dtype:
    resolved_device = device or infer_device()
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def resolve_dtype(cfg: dict, device: str) -> torch.dtype:
    dtype_name = str(cfg.get("dtype", "auto")).lower()

    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    if dtype_name == "auto":
        return infer_dtype(device)

    raise ValueError(
        f"Unknown dtype={dtype_name}. "
        "Choose from: auto, bf16, fp16, fp32."
    )


def save_adapter(model, config, output_dir: str, tag: str = "final") -> str:
    from .model import collect_adapter_weights

    save_path = os.path.join(output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    weights = collect_adapter_weights(model, config)
    torch.save(weights, os.path.join(save_path, "adapter_model.bin"))

    with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
        json.dump(config.export(), f, indent=2)

    logging.getLogger(__name__).info(f"Adapter saved to {save_path}")
    return save_path


def configure_file_logging(output_dir: str, filename: str = "train.log") -> str:
    log_path = os.path.abspath(os.path.join(output_dir, filename))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if os.path.abspath(handler.baseFilename) == log_path:
            return log_path
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)
    return log_path


def configure_external_log_levels() -> None:
    for logger_name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def collect_expert_sparsity_stats_with_mask(
    model: torch.nn.Module,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    active_experts_per_token_0_values = []
    active_experts_per_token_1e_3_values = []
    routing_sparsity_values = []
    flat_attention_mask = None
    if attention_mask is not None:
        flat_attention_mask = attention_mask.reshape(-1).to(dtype=torch.bool)

    for module in get_mixlora_moe_modules(model):
        runtime = getattr(module, "runtime_", None)
        if runtime is None:
            continue
        routing_weights = runtime.routing_weights
        if routing_weights is None:
            continue

        active_mask_0 = routing_weights > 0
        active_mask_1e_3 = routing_weights > 1e-3
        num_experts = active_mask_0.shape[-1]
        if flat_attention_mask is not None:
            if flat_attention_mask.numel() != active_mask_0.shape[0]:
                raise ValueError(
                    f"attention_mask has {flat_attention_mask.numel()} tokens, "
                    f"but routing_weights has {active_mask_0.shape[0]} tokens."
                )
            mask_device = flat_attention_mask.to(device=active_mask_0.device)
            active_mask_0 = active_mask_0[mask_device]
            active_mask_1e_3 = active_mask_1e_3[mask_device]
        if active_mask_0.numel() == 0:
            continue
        active_experts_per_token_0 = active_mask_0.float().sum(dim=-1).mean()
        active_experts_per_token_1e_3 = active_mask_1e_3.float().sum(dim=-1).mean()

        active_experts_per_token_0_values.append(active_experts_per_token_0)
        active_experts_per_token_1e_3_values.append(active_experts_per_token_1e_3)
        routing_sparsity_values.append(
            1.0 - (active_experts_per_token_0 / float(num_experts))
        )

    stats = {}
    if active_experts_per_token_0_values:
        stats["active_experts_per_token_0"] = float(
            torch.stack(active_experts_per_token_0_values).mean().detach().cpu()
        )
    if active_experts_per_token_1e_3_values:
        stats["active_experts_per_token_1e_3"] = float(
            torch.stack(active_experts_per_token_1e_3_values).mean().detach().cpu()
        )
    if routing_sparsity_values:
        stats["routing_sparsity"] = float(
            torch.stack(routing_sparsity_values).mean().detach().cpu()
        )
    return stats


def collect_layer_expert_usage_with_mask(
    model: torch.nn.Module,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[Dict[str, object]]:
    flat_attention_mask = None
    if attention_mask is not None:
        flat_attention_mask = attention_mask.reshape(-1).to(dtype=torch.bool)

    layer_usage = []
    for layer_idx, module in enumerate(get_mixlora_moe_modules(model)):
        runtime = getattr(module, "runtime_", None)
        if runtime is None:
            continue
        routing_weights = runtime.routing_weights
        if routing_weights is None:
            continue

        filtered_weights = routing_weights
        if flat_attention_mask is not None:
            if flat_attention_mask.numel() != routing_weights.shape[0]:
                raise ValueError(
                    f"attention_mask has {flat_attention_mask.numel()} tokens, "
                    f"but routing_weights has {routing_weights.shape[0]} tokens."
                )
            filtered_weights = routing_weights[
                flat_attention_mask.to(device=routing_weights.device)
            ]
        if filtered_weights.numel() == 0:
            continue

        layer_usage.append(
            {
                "layer_index": layer_idx,
                "num_tokens": int(filtered_weights.shape[0]),
                "routed_tokens_count": (
                    (filtered_weights > 0).float().sum(dim=0).detach().cpu().tolist()
                ),
            }
        )
    return layer_usage


def is_package_available(
    pkg_name: str, pkg_version: Optional[str] = None
) -> bool:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(
            pkg_version
        )
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()
