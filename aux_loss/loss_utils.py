from typing import Dict, Optional

import torch


def _zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _mean_or_zero(values, device: torch.device) -> torch.Tensor:
    if values:
        return torch.stack(values).mean()
    return _zero(device)

def _empty_router_stats_raw() -> Dict[str, float]:
    return {
        "active_assignments": 0.0,
        "total_assignments": 0.0,
        "token_rows": 0.0,
        "uncertainty_sum": 0.0,
        "uncertainty_count": 0.0,
        "exploration_coeff_sum": 0.0,
        "exploration_coeff_count": 0.0,
        "exploration_active_sum": 0.0,
        "exploration_active_count": 0.0,
        "top_evidence_sum": 0.0,
        "top_evidence_count": 0.0,
        "tail_evidence_sum": 0.0,
        "tail_evidence_count": 0.0,
        "tail_ratio_sum": 0.0,
        "tail_ratio_count": 0.0,
    }


def _truncated_exploration_coeff(
    uncertainty: torch.Tensor,
    threshold: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    rho = clip((u - tau) / (1 - tau), 0, 1)
    """
    scale = uncertainty.new_tensor(1.0 - threshold).clamp(min=eps)
    return ((uncertainty - threshold) / scale).clamp(min=0.0, max=1.0)


def _flatten_router_logits(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Support:
        [T, E] or [B, S, E]

    Return:
        [T, E]
    """
    return router_logits.reshape(-1, num_experts)


def _flatten_attention_mask(
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    attention_mask:
        [B, S], where 1 means valid token and 0 means padding.

    Return:
        [T] bool mask
    """
    if attention_mask is None:
        return None

    return attention_mask.reshape(-1).to(device=device).bool()


def _apply_attention_mask(
    router_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    router_logits:
        [T, E]

    attention_mask:
        [B, S]
    """
    if attention_mask is None:
        return router_logits

    mask = _flatten_attention_mask(attention_mask, router_logits.device)

    if mask is None:
        return router_logits

    if mask.numel() != router_logits.shape[0]:
        raise ValueError(
            f"attention_mask has {mask.numel()} tokens, "
            f"but router_logits has {router_logits.shape[0]} tokens."
        )

    return router_logits[mask]


def _apply_attention_mask_to_expert_outputs(
    expert_outputs: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    expert_outputs:
        [T, K, H]

    attention_mask:
        [B, S]
    """
    if attention_mask is None:
        return expert_outputs

    mask = _flatten_attention_mask(attention_mask, expert_outputs.device)

    if mask is None:
        return expert_outputs

    if mask.numel() != expert_outputs.shape[0]:
        raise ValueError(
            f"attention_mask has {mask.numel()} tokens, "
            f"but expert_outputs has {expert_outputs.shape[0]} tokens."
        )

    return expert_outputs[mask]


def _apply_flat_mask(
    tensor: torch.Tensor,
    flat_mask: Optional[torch.Tensor],
    tensor_name: str,
) -> torch.Tensor:
    if flat_mask is None:
        return tensor

    if flat_mask.numel() != tensor.shape[0]:
        raise ValueError(
            f"attention_mask has {flat_mask.numel()} tokens, "
            f"but {tensor_name} has {tensor.shape[0]} tokens."
        )

    return tensor[flat_mask.to(device=tensor.device)]
