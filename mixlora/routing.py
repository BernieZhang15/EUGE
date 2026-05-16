from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from aux_loss import _truncated_exploration_coeff


@dataclass
class RoutingResult:
    routing_weights: torch.Tensor
    selected_experts: Optional[torch.Tensor] = None
    router_probs: Optional[torch.Tensor] = None
    tsallis_entropy: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    exploration_coeff: Optional[torch.Tensor] = None
    exploration_mask: Optional[torch.Tensor] = None


def _topk_indices(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    return torch.topk(scores.detach(), top_k, dim=-1).indices


def _tsallis_entropy(probs: torch.Tensor, q: float) -> torch.Tensor:
    probs = probs.float()
    if abs(q - 1.0) < 1e-6:
        return -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return (1.0 - probs.pow(q).sum(dim=-1)) / (q - 1.0)


def _top_p_with_min_k(
    probs: torch.Tensor,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    cumulative_before = cumulative_probs - sorted_probs
    sorted_mask = cumulative_before < top_p
    sorted_mask[..., :top_k] = True

    full_mask = torch.zeros_like(sorted_mask)
    full_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    routing_weights = probs * full_mask.to(dtype=probs.dtype)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return routing_weights


def route_topk(
    router_logits: torch.Tensor,
    top_k: int,
    dtype: torch.dtype,
) -> RoutingResult:
    routing_weights = F.softmax(router_logits, dim=-1, dtype=dtype)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
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
    )


def route_remoe(
    router_logits: torch.Tensor,
    dtype: torch.dtype,
) -> RoutingResult:
    return RoutingResult(
        routing_weights=F.relu(router_logits).to(dtype),
    )


def route_dynmole(
    router_logits: torch.Tensor,
    top_k: int,
    dtype: torch.dtype,
    top_p: float,
    entropy_threshold: float,
    entropy_index: float,
) -> RoutingResult:
    router_probs = F.softmax(router_logits, dim=-1, dtype=dtype)
    tsallis_entropy = _tsallis_entropy(router_probs, entropy_index)
    high_entropy_mask = tsallis_entropy > entropy_threshold

    soft_routing = router_probs
    top_p_routing = _top_p_with_min_k(router_probs, top_p, top_k)
    routing_weights = torch.where(
        high_entropy_mask.unsqueeze(-1),
        soft_routing,
        top_p_routing,
    )

    return RoutingResult(
        routing_weights=routing_weights,
        selected_experts=_topk_indices(router_probs, top_k),
        router_probs=router_probs,
        tsallis_entropy=tsallis_entropy,
    )


def route_euge(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    u_threshold: float,
) -> RoutingResult:
    evidence = F.relu(router_logits)
    selected_experts = _topk_indices(evidence, top_k)

    evidence_sum = evidence.sum(dim=-1, keepdim=True)
    evidence_routing = torch.full_like(
        evidence,
        1.0 / float(num_experts),
    )
    has_evidence = evidence_sum > 0
    evidence_routing = torch.where(
        has_evidence,
        evidence / evidence_sum.clamp(min=1e-8),
        evidence_routing,
    )
    total_strength = float(num_experts) + evidence_sum
    uncertainty = float(num_experts) / (total_strength + 1e-8)

    exploration_coeff = _truncated_exploration_coeff(
        uncertainty,
        u_threshold,
    ).detach()
    routing_weights = (
        (1.0 - exploration_coeff) * evidence_routing
        + exploration_coeff * (1.0 / float(num_experts))
    )

    return RoutingResult(
        routing_weights=routing_weights.to(dtype),
        selected_experts=selected_experts,
        uncertainty=uncertainty,
        exploration_coeff=exploration_coeff,
        exploration_mask=exploration_coeff > 0,
    )


def compute_routing(
    router_logits: torch.Tensor,
    strategy: str,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    u_threshold: float,
    dynmole_top_p: float,
    dynmole_entropy_threshold: float,
    dynmole_entropy_index: float,
) -> RoutingResult:
    if strategy == "remoe":
        return route_remoe(router_logits, dtype)
    if strategy == "dynmole":
        return route_dynmole(
            router_logits=router_logits,
            top_k=top_k,
            dtype=dtype,
            top_p=dynmole_top_p,
            entropy_threshold=dynmole_entropy_threshold,
            entropy_index=dynmole_entropy_index,
        )
    if strategy == "EUGE":
        return route_euge(
            router_logits=router_logits,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
            u_threshold=u_threshold,
        )
    return route_topk(
        router_logits=router_logits,
        top_k=top_k,
        dtype=dtype,
    )
