from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from aux_loss.loss_utils import build_euge_intermediates


@dataclass
class RoutingResult:
    routing_weights: torch.Tensor
    selected_experts: Optional[torch.Tensor] = None
    router_probs: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    exploration_coeff: Optional[torch.Tensor] = None
    exploration_mask: Optional[torch.Tensor] = None
    evidence: Optional[torch.Tensor] = None
    evidence_sum: Optional[torch.Tensor] = None
    top_evidence: Optional[torch.Tensor] = None
    tail_evidence: Optional[torch.Tensor] = None


def _topk_indices(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    return torch.topk(scores.detach(), top_k, dim=-1).indices


def route_topk(
    router_logits: torch.Tensor,
    top_k: int,
    dtype: torch.dtype,
) -> RoutingResult:
    router_probs = F.softmax(router_logits, dim=-1, dtype=dtype)
    routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    full_weights = torch.zeros(
        router_logits.shape[0],
        router_logits.shape[1],
        dtype=dtype,
        device=router_logits.device,
    )
    full_weights.scatter_(1, selected_experts, routing_weights)
    return RoutingResult(
        routing_weights=full_weights,
        selected_experts=selected_experts,
        router_probs=router_probs,
    )


def route_loss_free(
    router_logits: torch.Tensor,
    top_k: int,
    dtype: torch.dtype,
    expert_bias: torch.Tensor,
) -> RoutingResult:
    router_probs_fp32 = torch.sigmoid(router_logits.float())
    biased_scores = router_probs_fp32 + expert_bias.to(
        device=router_probs_fp32.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    selected_experts = torch.topk(biased_scores, top_k, dim=-1).indices
    selected_weights = router_probs_fp32.gather(dim=-1, index=selected_experts)
    selected_weights /= selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    selected_weights = selected_weights.to(dtype)

    full_weights = torch.zeros(
        router_logits.shape[0],
        router_logits.shape[1],
        dtype=dtype,
        device=router_logits.device,
    )
    full_weights.scatter_(1, selected_experts, selected_weights)
    return RoutingResult(
        routing_weights=full_weights,
        selected_experts=selected_experts,
        router_probs=router_probs_fp32.to(dtype),
    )


def route_euge(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    u_threshold: float,
    inference_mode: str = "dense",
) -> RoutingResult:
    intermediates = build_euge_intermediates(
        router_logits=router_logits,
        top_k=top_k,
        num_experts=num_experts,
        u_threshold=u_threshold,
    )
    dense_routing_weights = (
        (1.0 - intermediates.exploration_coeff) * intermediates.evidence_routing
        + intermediates.exploration_coeff * (1.0 / float(num_experts))
    )
    if inference_mode == "sparse":
        selected_weights = dense_routing_weights.gather(
            dim=-1,
            index=intermediates.selected_experts,
        )
        routing_weights = torch.zeros_like(dense_routing_weights)
        routing_weights.scatter_(1, intermediates.selected_experts, selected_weights)
        exploration_coeff = intermediates.exploration_coeff
        exploration_mask = intermediates.exploration_coeff > 0
    else:
        routing_weights = dense_routing_weights
        exploration_coeff = intermediates.exploration_coeff
        exploration_mask = intermediates.exploration_coeff > 0

    return RoutingResult(
        routing_weights=routing_weights.to(dtype),
        selected_experts=intermediates.selected_experts,
        uncertainty=intermediates.uncertainty,
        exploration_coeff=exploration_coeff,
        exploration_mask=exploration_mask,
        evidence=intermediates.evidence,
        evidence_sum=intermediates.evidence_sum,
        top_evidence=intermediates.top_evidence,
        tail_evidence=intermediates.tail_evidence,
    )


def compute_routing(
    router_logits: torch.Tensor,
    strategy: str,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    loss_free_bias: torch.Tensor,
    u_threshold: float,
    inference_mode: str = "dense",
) -> RoutingResult:
    if strategy == "loss-free":
        return route_loss_free(
            router_logits=router_logits,
            top_k=top_k,
            dtype=dtype,
            expert_bias=loss_free_bias,
        )
    if strategy == "EUGE":
        return route_euge(
            router_logits=router_logits,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
            u_threshold=u_threshold,
            inference_mode=inference_mode,
        )
    return route_topk(
        router_logits=router_logits,
        top_k=top_k,
        dtype=dtype,
    )
