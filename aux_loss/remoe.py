from typing import Optional

import torch

from .loss_utils import _flatten_attention_mask, _flatten_router_logits


def remoe_regularization_loss(
    routing_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ReMoE load-balancing refined L1 regularization.

    This follows the official implementation: use the raw ReLU router outputs
    as expert weights, and weight them by each expert's activation ratio
    relative to the target top-k compute budget.
    """
    if coef == 0.0:
        return routing_weights.new_zeros(())

    routing_weights = _flatten_router_logits(routing_weights, num_experts)
    active_mask = routing_weights > 0

    if attention_mask is not None:
        flat_mask = _flatten_attention_mask(attention_mask, routing_weights.device)
        if flat_mask.numel() != routing_weights.shape[0]:
            raise ValueError(
                f"attention_mask has {flat_mask.numel()} tokens, "
                f"but routing_weights has {routing_weights.shape[0]} tokens."
            )
        routing_weights = routing_weights[flat_mask]
        active_mask = active_mask[flat_mask]

    if routing_weights.numel() == 0:
        return routing_weights.new_zeros(())

    routing_weights = routing_weights.float()
    probs_per_expert = routing_weights.mean(dim=0)
    tokens_per_expert = active_mask.float().sum(dim=0)
    desired_assignments = max(routing_weights.shape[0] * top_k, 1)
    expert_fraction = tokens_per_expert / float(desired_assignments)
    loss = num_experts * torch.sum(probs_per_expert * expert_fraction)
    return coef * loss
