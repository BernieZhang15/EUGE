import logging
import math
from typing import Dict

import torch
import torch.nn as nn

from .config import MixLoraConfig
from .model import (
    MixLoraSparseMoe,
    _gate_bias_key,
    _gate_key,
    _lora_key,
    inject_adapter_in_model,
)

logger = logging.getLogger(__name__)


def _init_gate(
    hidden_size: int,
    num_experts: int,
    init_range: float,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    gate = torch.empty(num_experts, hidden_size, dtype=dtype, device=device)
    nn.init.normal_(gate, std=init_range)
    return gate


def _init_gate_bias(
    num_experts: int,
    init_value: float,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    return torch.full((num_experts,), init_value, dtype=dtype, device=device)


def build_mixlora_model(
    base_model_name: str,
    mixlora_config: MixLoraConfig,
    device: str,
    dtype: torch.dtype,
):
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    mixlora_config.model_type_ = model.config.model_type
    mixlora_config.dtype_ = dtype

    weights: Dict[str, torch.Tensor] = {}
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    for layer_idx in range(num_layers):
        weights[_gate_key(layer_idx)] = _init_gate(
            hidden_size,
            mixlora_config.num_experts_,
            mixlora_config.router_init_range_,
            dtype,
            device,
        )
        if mixlora_config.router_bias_init_ != 0.0:
            weights[_gate_bias_key(layer_idx)] = _init_gate_bias(
                mixlora_config.num_experts_,
                mixlora_config.router_bias_init_,
                dtype,
                device,
            )
        mlp = model.model.layers[layer_idx].mlp
        for proj_name, inject in mixlora_config.target_modules_.items():
            if not inject or not hasattr(mlp, proj_name):
                continue
            proj_layer = getattr(mlp, proj_name)
            in_dim, out_dim = proj_layer.in_features, proj_layer.out_features
            for expert_idx in range(mixlora_config.num_experts_):
                lora_A = torch.empty(
                    mixlora_config.lora_r_,
                    in_dim,
                    dtype=dtype,
                    device=device,
                )
                lora_B = torch.zeros(
                    out_dim,
                    mixlora_config.lora_r_,
                    dtype=dtype,
                    device=device,
                )
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_A")] = lora_A
                weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_B")] = lora_B

    logger.info(f"Injecting MixLoRA adapters (strategy={mixlora_config.routing_strategy_}) ...")
    inject_adapter_in_model(model, mixlora_config, weights)
    model._mixlora_moe_modules = [
        module for module in model.modules() if isinstance(module, MixLoraSparseMoe)
    ]

    for name, param in model.named_parameters():
        param.requires_grad = any(
            key in name for key in ("lora_A", "lora_B", "moe_gate.weight", "moe_gate.bias")
        )

    for module in model.modules():
        if isinstance(module, MixLoraSparseMoe) and module.gate_ is not None:
            module.gate_.requires_grad_(True)
            if module.gate_bias_ is not None:
                module.gate_bias_.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model
