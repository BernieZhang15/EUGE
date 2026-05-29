from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

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


@dataclass
class ExpertForwardResult:
    final_hidden_states: torch.Tensor
    ortho_outputs: Optional[torch.Tensor] = None
    ortho_active_mask: Optional[torch.Tensor] = None


def _prepare_ortho_buffers(
    routing_strategy: str,
    num_experts: int,
    hidden_states: torch.Tensor,
    dtype: torch.dtype,
    routing_weights: torch.Tensor,
    selected_experts: Optional[torch.Tensor],
    collect_ortho: bool,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
]:
    if not collect_ortho:
        return None, None, None, None

    num_tokens, hidden_dim = hidden_states.shape
    if routing_strategy == "EUGE":
        if selected_experts is None or selected_experts.numel() == 0:
            return None, None, None, None
        max_active_experts = selected_experts.shape[-1]
        if max_active_experts <= 1:
            return None, None, None, None
        ortho_outputs = torch.zeros(
            (num_tokens, max_active_experts, hidden_dim),
            dtype=dtype,
            device=hidden_states.device,
        )
        ortho_active_mask = torch.ones(
            (num_tokens, max_active_experts),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        selected_positions = [
            torch.where(selected_experts == expert_idx)
            for expert_idx in range(num_experts)
        ]
        return ortho_outputs, ortho_active_mask, None, selected_positions

    active_mask = routing_weights > 0
    active_counts = active_mask.sum(dim=-1)
    if active_counts.numel() == 0:
        return None, None, None, None
    max_active_experts = int(active_counts.max().item())
    if max_active_experts <= 1:
        return None, None, None, None

    ortho_outputs = torch.zeros(
        (num_tokens, max_active_experts, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    ortho_active_mask = torch.zeros(
        (num_tokens, max_active_experts),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    active_slots = active_mask.long().cumsum(dim=-1) - 1
    return ortho_outputs, ortho_active_mask, active_slots, None


def _scatter_ortho_outputs(
    routing_strategy: str,
    expert_idx: int,
    expert_output: torch.Tensor,
    top_x: Optional[torch.Tensor],
    dense_all_experts: bool,
    ortho_outputs: Optional[torch.Tensor],
    ortho_active_mask: Optional[torch.Tensor],
    active_slots: Optional[torch.Tensor],
    selected_positions: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    if ortho_outputs is None or ortho_active_mask is None:
        return

    if routing_strategy == "EUGE":
        if selected_positions is None:
            return
        token_positions, slot_positions = selected_positions[expert_idx]
        if token_positions.numel() == 0:
            return
        if dense_all_experts:
            ortho_outputs[token_positions, slot_positions] = expert_output[token_positions]
            return
        if top_x is None or top_x.numel() == 0:
            return
        local_positions = torch.searchsorted(top_x, token_positions)
        valid_positions = local_positions < top_x.numel()
        if valid_positions.any():
            valid_local = local_positions[valid_positions]
            valid_token = token_positions[valid_positions]
            valid_slot = slot_positions[valid_positions]
            valid_positions = top_x[valid_local] == valid_token
            if valid_positions.any():
                ortho_outputs[
                    valid_token[valid_positions],
                    valid_slot[valid_positions],
                ] = expert_output[valid_local[valid_positions]]
        return

    if top_x is None or top_x.numel() == 0 or active_slots is None:
        return
    slot_x = active_slots[top_x, expert_idx]
    ortho_outputs[top_x, slot_x] = expert_output
    ortho_active_mask[top_x, slot_x] = True


def _llama_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
    routing_weights: torch.Tensor,
    routing_strategy: str,
    selected_experts: Optional[torch.Tensor],
    collect_ortho: bool,
    dense_all_experts: bool,
) -> ExpertForwardResult:
    x = hidden_states.to(input_dtype)
    common_gate = base_layer.gate_proj(x).to(dtype)
    common_up = base_layer.up_proj(x).to(dtype)
    num_tokens, hidden_dim = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    gate_experts = get_expert_group("gate_proj")
    up_experts = get_expert_group("up_proj")
    down_experts = get_expert_group("down_proj")
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    ortho_outputs, ortho_active_mask, active_slots, selected_positions = _prepare_ortho_buffers(
        routing_strategy=routing_strategy,
        num_experts=num_experts,
        hidden_states=hidden_states,
        dtype=dtype,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        collect_ortho=collect_ortho,
    )

    for expert_idx in range(num_experts):
        top_x = None if dense_all_experts else expert_token_indices[expert_idx]
        if not dense_all_experts and top_x.numel() == 0:
            continue
        hidden_slice = hidden_states if dense_all_experts else _slice_tensor(hidden_states, top_x, dtype)

        lora_gate = gate_experts[expert_idx]
        lora_up = up_experts[expert_idx]
        lora_down = down_experts[expert_idx]
        gate_slice = common_gate if dense_all_experts else _slice_tensor(common_gate, top_x, dtype)
        up_slice = common_up if dense_all_experts else _slice_tensor(common_up, top_x, dtype)

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
        expert_output = (
            lora_down.lora_forward(down, act_result) if lora_down is not None else down
        )
        if dense_all_experts:
            final_hidden_states += expert_output * routing_weights[:, expert_idx, None]
        else:
            final_hidden_states.index_add_(
                0,
                top_x,
                expert_output * routing_weights[top_x, expert_idx, None],
            )
        _scatter_ortho_outputs(
            routing_strategy=routing_strategy,
            expert_idx=expert_idx,
            expert_output=expert_output,
            top_x=top_x,
            dense_all_experts=dense_all_experts,
            ortho_outputs=ortho_outputs,
            ortho_active_mask=ortho_active_mask,
            active_slots=active_slots,
            selected_positions=selected_positions,
        )
    return ExpertForwardResult(
        final_hidden_states=final_hidden_states,
        ortho_outputs=ortho_outputs,
        ortho_active_mask=ortho_active_mask,
    )


def _phi_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
    routing_weights: torch.Tensor,
    routing_strategy: str,
    selected_experts: Optional[torch.Tensor],
    collect_ortho: bool,
    dense_all_experts: bool,
) -> ExpertForwardResult:
    x = hidden_states.to(input_dtype)
    common_fc1 = base_layer.fc1(x).to(dtype)
    num_tokens, hidden_dim = hidden_states.shape
    fc1_experts = get_expert_group("fc1")
    fc2_experts = get_expert_group("fc2")
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    ortho_outputs, ortho_active_mask, active_slots, selected_positions = _prepare_ortho_buffers(
        routing_strategy=routing_strategy,
        num_experts=num_experts,
        hidden_states=hidden_states,
        dtype=dtype,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        collect_ortho=collect_ortho,
    )

    for expert_idx in range(num_experts):
        top_x = None if dense_all_experts else expert_token_indices[expert_idx]
        if not dense_all_experts and top_x.numel() == 0:
            continue
        hidden_slice = hidden_states if dense_all_experts else _slice_tensor(hidden_states, top_x, dtype)

        lora_fc1 = fc1_experts[expert_idx]
        lora_fc2 = fc2_experts[expert_idx]
        fc1_slice = common_fc1 if dense_all_experts else _slice_tensor(common_fc1, top_x, dtype)

        fc1_out = (
            lora_fc1.lora_forward(fc1_slice, hidden_slice)
            if lora_fc1 is not None else fc1_slice
        )
        act_result = act_fn(fc1_out)
        fc2 = base_layer.fc2(act_result.to(input_dtype)).to(dtype)
        expert_output = (
            lora_fc2.lora_forward(fc2, act_result) if lora_fc2 is not None else fc2
        )
        if dense_all_experts:
            final_hidden_states += expert_output * routing_weights[:, expert_idx, None]
        else:
            final_hidden_states.index_add_(
                0,
                top_x,
                expert_output * routing_weights[top_x, expert_idx, None],
            )
        _scatter_ortho_outputs(
            routing_strategy=routing_strategy,
            expert_idx=expert_idx,
            expert_output=expert_output,
            top_x=top_x,
            dense_all_experts=dense_all_experts,
            ortho_outputs=ortho_outputs,
            ortho_active_mask=ortho_active_mask,
            active_slots=active_slots,
            selected_positions=selected_positions,
        )
    return ExpertForwardResult(
        final_hidden_states=final_hidden_states,
        ortho_outputs=ortho_outputs,
        ortho_active_mask=ortho_active_mask,
    )


def _phi3_expert_forward(
    base_layer: nn.Module,
    get_expert_group: Callable[[str], List[Optional[LoraLinear]]],
    num_experts: int,
    act_fn,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
    routing_weights: torch.Tensor,
    routing_strategy: str,
    selected_experts: Optional[torch.Tensor],
    collect_ortho: bool,
    dense_all_experts: bool,
) -> ExpertForwardResult:
    x = hidden_states.to(input_dtype)
    common_gate_up = base_layer.gate_up_proj(x).to(dtype)
    num_tokens, hidden_dim = hidden_states.shape
    gate_up_experts = get_expert_group("gate_up_proj")
    down_experts = get_expert_group("down_proj")
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    ortho_outputs, ortho_active_mask, active_slots, selected_positions = _prepare_ortho_buffers(
        routing_strategy=routing_strategy,
        num_experts=num_experts,
        hidden_states=hidden_states,
        dtype=dtype,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        collect_ortho=collect_ortho,
    )

    for expert_idx in range(num_experts):
        top_x = None if dense_all_experts else expert_token_indices[expert_idx]
        if not dense_all_experts and top_x.numel() == 0:
            continue
        hidden_slice = hidden_states if dense_all_experts else _slice_tensor(hidden_states, top_x, dtype)

        lora_gate_up = gate_up_experts[expert_idx]
        lora_down = down_experts[expert_idx]
        gate_up_slice = (
            common_gate_up if dense_all_experts else _slice_tensor(common_gate_up, top_x, dtype)
        )

        gate_up_states = (
            lora_gate_up.lora_forward(gate_up_slice, hidden_slice)
            if lora_gate_up is not None else gate_up_slice
        )
        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        act_result = up_states * act_fn(gate_states)
        down = base_layer.down_proj(act_result.to(input_dtype)).to(dtype)
        expert_output = (
            lora_down.lora_forward(down, act_result) if lora_down is not None else down
        )
        if dense_all_experts:
            final_hidden_states += expert_output * routing_weights[:, expert_idx, None]
        else:
            final_hidden_states.index_add_(
                0,
                top_x,
                expert_output * routing_weights[top_x, expert_idx, None],
            )
        _scatter_ortho_outputs(
            routing_strategy=routing_strategy,
            expert_idx=expert_idx,
            expert_output=expert_output,
            top_x=top_x,
            dense_all_experts=dense_all_experts,
            ortho_outputs=ortho_outputs,
            ortho_active_mask=ortho_active_mask,
            active_slots=active_slots,
            selected_positions=selected_positions,
        )
    return ExpertForwardResult(
        final_hidden_states=final_hidden_states,
        ortho_outputs=ortho_outputs,
        ortho_active_mask=ortho_active_mask,
    )


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
