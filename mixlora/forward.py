from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .lora_linear import LoraLinear


def _slice_tensor(
    data: torch.Tensor,
    indices: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if data.dtype == dtype:
        return data[indices]
    return data[indices].to(dtype)


def _llama_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
) -> List[torch.Tensor]:
    x = hidden_states.to(input_dtype)
    common_gate = base_layer.gate_proj(x).to(dtype)
    common_up = base_layer.up_proj(x).to(dtype)
    hidden_dim = hidden_states.shape[-1]
    gate_experts = get_expert_group("gate_proj")
    up_experts = get_expert_group("up_proj")
    down_experts = get_expert_group("down_proj")

    final_expert_states = []
    for expert_idx in range(num_experts):
        top_x = expert_token_indices[expert_idx]
        if top_x.numel() == 0:
            final_expert_states.append(hidden_states.new_zeros((0, hidden_dim)))
            continue
        hidden_slice = _slice_tensor(hidden_states, top_x, dtype)

        lora_gate = gate_experts[expert_idx]
        lora_up = up_experts[expert_idx]
        lora_down = down_experts[expert_idx]
        gate_slice = _slice_tensor(common_gate, top_x, dtype)
        up_slice = _slice_tensor(common_up, top_x, dtype)

        gate_states = (
            lora_gate.lora_forward(gate_slice, hidden_slice)
            if lora_gate is not None else gate_slice
        )
        up_states = (
            lora_up.lora_forward(up_slice, hidden_slice)
            if lora_up is not None else up_slice
        )
        act_result = act_fn(gate_states) * up_states
        down = base_layer.down_proj(act_result.to(input_dtype)).to(dtype)
        final_expert_states.append(
            lora_down.lora_forward(down, act_result) if lora_down is not None else down
        )
    return final_expert_states


def _phi_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
) -> List[torch.Tensor]:
    x = hidden_states.to(input_dtype)
    common_fc1 = base_layer.fc1(x).to(dtype)
    hidden_dim = hidden_states.shape[-1]
    fc1_experts = get_expert_group("fc1")
    fc2_experts = get_expert_group("fc2")

    final_expert_states = []
    for expert_idx in range(num_experts):
        top_x = expert_token_indices[expert_idx]
        if top_x.numel() == 0:
            final_expert_states.append(hidden_states.new_zeros((0, hidden_dim)))
            continue
        hidden_slice = _slice_tensor(hidden_states, top_x, dtype)

        lora_fc1 = fc1_experts[expert_idx]
        lora_fc2 = fc2_experts[expert_idx]
        fc1_slice = _slice_tensor(common_fc1, top_x, dtype)

        fc1_out = (
            lora_fc1.lora_forward(fc1_slice, hidden_slice)
            if lora_fc1 is not None else fc1_slice
        )
        act_result = act_fn(fc1_out)
        fc2 = base_layer.fc2(act_result.to(input_dtype)).to(dtype)
        final_expert_states.append(
            lora_fc2.lora_forward(fc2, act_result) if lora_fc2 is not None else fc2
        )
    return final_expert_states


def _phi3_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
) -> List[torch.Tensor]:
    x = hidden_states.to(input_dtype)
    common_gate_up = base_layer.gate_up_proj(x).to(dtype)
    hidden_dim = hidden_states.shape[-1]
    gate_up_experts = get_expert_group("gate_up_proj")
    down_experts = get_expert_group("down_proj")

    final_expert_states = []
    for expert_idx in range(num_experts):
        top_x = expert_token_indices[expert_idx]
        if top_x.numel() == 0:
            final_expert_states.append(hidden_states.new_zeros((0, hidden_dim)))
            continue
        hidden_slice = _slice_tensor(hidden_states, top_x, dtype)

        lora_gate_up = gate_up_experts[expert_idx]
        lora_down = down_experts[expert_idx]
        gate_up_slice = _slice_tensor(common_gate_up, top_x, dtype)

        gate_up_states = (
            lora_gate_up.lora_forward(gate_up_slice, hidden_slice)
            if lora_gate_up is not None else gate_up_slice
        )
        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        act_result = up_states * act_fn(gate_states)
        down = base_layer.down_proj(act_result.to(input_dtype)).to(dtype)
        final_expert_states.append(
            lora_down.lora_forward(down, act_result) if lora_down is not None else down
        )
    return final_expert_states


_COMPATIBLE_MODEL_TYPES = {
    "llama": _llama_expert_forward,
    "gemma": _llama_expert_forward,
    "gemma2": _llama_expert_forward,
    "qwen2": _llama_expert_forward,
    "mistral": _llama_expert_forward,
    "phi": _phi_expert_forward,
    "phi3": _phi3_expert_forward,
}


def resolve_expert_forward(model_type: str):
    if model_type not in _COMPATIBLE_MODEL_TYPES:
        raise NotImplementedError(
            f"Unsupported model type '{model_type}'. "
            f"Supported: {list(_COMPATIBLE_MODEL_TYPES)}"
        )
    return _COMPATIBLE_MODEL_TYPES[model_type]
