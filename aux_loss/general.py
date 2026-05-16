from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .loss_utils import (
    _apply_attention_mask,
    _apply_attention_mask_to_expert_outputs,
    _flatten_attention_mask,
    _flatten_router_logits,
    _truncated_exploration_coeff,
)


def topk_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Switch-style top-k load balancing loss.

    L_aux = coef * E * sum_i f_i * P_i

    where:
        f_i = fraction of top-k assignments routed to expert i
        P_i = average router probability assigned to expert i
    """

    if coef == 0.0:
        return router_logits.new_zeros(())

    router_logits = _flatten_router_logits(router_logits, num_experts)
    router_logits = _apply_attention_mask(router_logits, attention_mask)

    if router_logits.numel() == 0:
        return router_logits.new_zeros(())

    routing_probs = torch.softmax(router_logits.float(), dim=-1)
    _, selected = torch.topk(routing_probs, top_k, dim=-1)
    expert_mask = F.one_hot(selected, num_classes=num_experts).float()

    tokens_per_expert = expert_mask.sum(dim=(0, 1))
    expert_fraction = tokens_per_expert / tokens_per_expert.sum().clamp(min=1e-6)
    router_prob_per_expert = routing_probs.mean(dim=0)
    loss = num_experts * torch.sum(expert_fraction * router_prob_per_expert)

    return coef * loss

def discriminative_routing_loss(
    routing_weights: torch.Tensor,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Variance-based router loss aligned with the released implementation of
    "Advancing Expert Specialization for Better MoE".

    It computes the variance of each token's routing weight vector across
    experts, then averages over tokens:

        L_disc = - mean_i Var_j(s_ij)

    The routing weights can come from softmax top-k routing or custom
    evidence-based routing. The key requirement is that they represent the
    per-token expert weight vector after routing.
    """
    if coef == 0.0:
        return routing_weights.new_zeros(())

    num_experts = routing_weights.shape[-1]
    routing_weights = _flatten_router_logits(routing_weights, num_experts)
    routing_weights = _apply_attention_mask(routing_weights, attention_mask)

    if routing_weights.numel() == 0:
        return routing_weights.new_zeros(())

    routing_weights = routing_weights.float()
    variance = torch.var(routing_weights, dim=-1).mean()
    return -coef * variance


def expert_output_orthogonality_loss(
    expert_outputs: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Orthogonality loss on per-token expert outputs.

    Matches Eq. (6) in "Advancing Expert Specialization for Better MoE":
      - input shape: [T, A, H], where A is the compact active-expert axis
      - active_mask marks which slots are active for each token
      - for each token, sum the squared norm of the projection of each
        active expert output onto every other active expert output
      - average over active ordered expert pairs for stable scale
    """
    if coef == 0.0:
        return expert_outputs.new_zeros(())

    if expert_outputs.ndim != 3:
        raise ValueError(
            f"expert_outputs must have shape [T, E, H], got {tuple(expert_outputs.shape)}"
        )

    expert_outputs = _apply_attention_mask_to_expert_outputs(expert_outputs, attention_mask)
    if active_mask is None:
        active_mask = torch.ones(
            expert_outputs.shape[:2],
            dtype=torch.bool,
            device=expert_outputs.device,
        )
    else:
        if active_mask.ndim != 2:
            raise ValueError(
                f"active_mask must have shape [T, E], got {tuple(active_mask.shape)}"
            )
        if attention_mask is not None:
            flat_attention_mask = _flatten_attention_mask(attention_mask, expert_outputs.device)
            if flat_attention_mask is not None:
                if flat_attention_mask.numel() != active_mask.shape[0]:
                    raise ValueError(
                        f"attention_mask has {flat_attention_mask.numel()} tokens, "
                        f"but active_mask has {active_mask.shape[0]} tokens."
                    )
                active_mask = active_mask[flat_attention_mask]
        active_mask = active_mask.to(device=expert_outputs.device, dtype=torch.bool)

    if expert_outputs.numel() == 0:
        return expert_outputs.new_zeros(())

    num_experts = expert_outputs.shape[1]
    if num_experts <= 1:
        return expert_outputs.new_zeros(())

    expert_outputs = expert_outputs.float()
    reference = expert_outputs[:, None, :, :]
    current = expert_outputs[:, :, None, :]

    dot = (current * reference).sum(dim=-1)
    reference_norm_sq = reference.square().sum(dim=-1)

    projection_norm_sq = dot.square() * reference_norm_sq / reference_norm_sq.add(1e-6).square()

    pair_mask = (
        active_mask[:, :, None]
        & active_mask[:, None, :]
        & ~torch.eye(num_experts, device=expert_outputs.device, dtype=torch.bool).unsqueeze(0)
    )
    active_pair_count = pair_mask.sum()
    if active_pair_count.item() == 0:
        return expert_outputs.new_zeros(())

    total_loss = projection_norm_sq.masked_select(pair_mask).sum()
    total_loss = total_loss / active_pair_count.to(dtype=expert_outputs.dtype)

    return coef * total_loss


def _build_compact_outputs_for_selected_experts(
    selected_experts: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    expert_states: List[torch.Tensor],
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    max_active_experts = selected_experts.shape[-1]

    compact_outputs = torch.zeros(
        (num_tokens, max_active_experts, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    compact_active_mask = torch.ones(
        (num_tokens, max_active_experts),
        dtype=torch.bool,
        device=hidden_states.device,
    )

    for expert_idx in range(num_experts):
        top_x = expert_token_indices[expert_idx]
        if top_x.numel() == 0:
            continue

        token_positions, slot_positions = torch.where(selected_experts == expert_idx)
        if token_positions.numel() == 0:
            continue

        if top_x.numel() == num_tokens:
            compact_outputs[token_positions, slot_positions] = expert_states[expert_idx][
                token_positions
            ]
            continue

        token_to_local = torch.full(
            (num_tokens,),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        token_to_local[top_x] = torch.arange(
            top_x.numel(),
            device=hidden_states.device,
        )
        local_positions = token_to_local[token_positions]
        valid_positions = local_positions >= 0
        if not valid_positions.any():
            continue

        compact_outputs[
            token_positions[valid_positions],
            slot_positions[valid_positions],
        ] = expert_states[expert_idx][local_positions[valid_positions]]

    return compact_outputs, compact_active_mask


def _build_compact_outputs_from_active_mask(
    routing_weights: torch.Tensor,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    expert_states: List[torch.Tensor],
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    active_mask = routing_weights > 0
    active_counts = active_mask.sum(dim=-1)
    max_active_experts = int(active_counts.max().item())
    active_slots = active_mask.long().cumsum(dim=-1) - 1

    compact_outputs = torch.zeros(
        (num_tokens, max_active_experts, hidden_dim),
        dtype=dtype,
        device=hidden_states.device,
    )
    compact_active_mask = torch.zeros(
        (num_tokens, max_active_experts),
        dtype=torch.bool,
        device=hidden_states.device,
    )

    for expert_idx, top_x in enumerate(expert_token_indices):
        if top_x.numel() == 0:
            continue
        slot_x = active_slots[top_x, expert_idx]
        compact_outputs[top_x, slot_x] = expert_states[expert_idx]
        compact_active_mask[top_x, slot_x] = True

    return compact_outputs, compact_active_mask


def compute_moe_ortho_loss(
    training: bool,
    routing_strategy: str,
    coef: float,
    selected_experts: Optional[torch.Tensor],
    num_experts: int,
    dtype: torch.dtype,
    expert_token_indices: List[torch.Tensor],
    routing_weights: torch.Tensor,
    expert_states: List[torch.Tensor],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not training or coef <= 0.0 or routing_strategy == "remoe":
        return hidden_states.new_zeros(())

    if routing_strategy == "EUGE":
        if selected_experts is None or selected_experts.numel() == 0:
            return hidden_states.new_zeros(())
        max_active_experts = selected_experts.shape[-1]
        if max_active_experts <= 1:
            return hidden_states.new_zeros(())
        compact_outputs, compact_active_mask = _build_compact_outputs_for_selected_experts(
            selected_experts=selected_experts,
            num_experts=num_experts,
            dtype=dtype,
            expert_token_indices=expert_token_indices,
            expert_states=expert_states,
            hidden_states=hidden_states,
        )
    else:
        active_counts = (routing_weights > 0).sum(dim=-1)
        if active_counts.numel() == 0:
            return hidden_states.new_zeros(())
        max_active_experts = int(active_counts.max().item())
        if max_active_experts <= 1:
            return hidden_states.new_zeros(())
        compact_outputs, compact_active_mask = _build_compact_outputs_from_active_mask(
            routing_weights=routing_weights,
            dtype=dtype,
            expert_token_indices=expert_token_indices,
            expert_states=expert_states,
            hidden_states=hidden_states,
        )

    return expert_output_orthogonality_loss(
        compact_outputs,
        active_mask=compact_active_mask,
        coef=coef,
    )


def evidential_sparsity_loss(
    router_logits: torch.Tensor,
    top_k: int,
    u_threshold: float,
    lambda_sparse: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Evidential sparsity loss for uncertainty-guided evidential routing.

    Steps:
      1. e = relu(router_logits)
      2. Build detached top-k potential-expert mask from e
      3. alpha_neg = 1 + (1 - mask) * e
      4. KL[Dir(alpha_neg) || Dir(ones)]
      5. Exploration coefficient rho = sg(clip((u - tau) / (1 - tau), 0, 1))
      6. Average over valid tokens
      7. Return lambda_sparse * mean(loss_per_token)
    """
    if lambda_sparse == 0.0:
        return router_logits.new_zeros(())

    num_experts = router_logits.shape[-1]
    router_logits = _flatten_router_logits(router_logits, num_experts)
    router_logits = _apply_attention_mask(router_logits, attention_mask)

    if router_logits.numel() == 0:
        return router_logits.new_zeros(())

    e = torch.relu(router_logits.float())
    topk_idx = torch.topk(e.detach(), k=top_k, dim=-1).indices

    mask = torch.zeros_like(e)
    mask.scatter_(-1, topk_idx, 1.0)

    alpha = 1.0 + (1.0 - mask) * e

    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    log_B_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    log_B_ones = -torch.lgamma(
        torch.tensor(float(num_experts), device=alpha.device, dtype=alpha.dtype)
    )

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)
    correction = ((alpha - 1.0) * (digamma_alpha - digamma_sum_alpha)).sum(
        dim=-1,
        keepdim=True,
    )

    kl = (log_B_alpha + log_B_ones + correction).squeeze(-1)

    evidence_sum = e.sum(dim=-1)
    uncertainty = num_experts / (num_experts + evidence_sum + eps)
    rho = _truncated_exploration_coeff(uncertainty, u_threshold, eps).detach()
    loss_per_token = (1.0 - rho) * kl

    return lambda_sparse * loss_per_token.mean()


def evidence_calibration_loss(
    router_logits: torch.Tensor,
    top_k: int,
    u_threshold: float,
    coef: float,
    eta: float,
    eps: float = 1e-8,
    attention_mask: Optional[torch.Tensor] = None,
    task_valid_mask: Optional[torch.Tensor] = None,
    task_loss_per_token: Optional[torch.Tensor] = None,
    loss_min: float = 0.0,
    loss_max: float = 3.0,
) -> torch.Tensor:
    """
    Evidence calibration loss for an evidential MoE router.

    L_cal = clipped_task_loss_detached * log(S + eps) + eta * K / (S_seek + eps)

    where:
      - e = relu(router_logits)
      - S = K + sum(e)
      - S_seek = K + E_top + rho * E_tail
      - E_top uses a detached top-k potential-expert mask
      - rho is the detached exploration coefficient
    """
    if coef == 0.0:
        return router_logits.new_zeros(())
    if top_k is None:
        raise ValueError("evidence_calibration_loss requires top_k.")

    num_experts = router_logits.shape[-1]
    router_logits = _flatten_router_logits(router_logits, num_experts)

    if task_loss_per_token is None:
        raise ValueError("evidence_calibration_loss requires task_loss_per_token.")

    task_loss = task_loss_per_token.reshape(-1).to(
        device=router_logits.device,
        dtype=router_logits.dtype,
    )
    if task_loss.numel() != router_logits.shape[0]:
        raise ValueError(
            f"task_loss_per_token has {task_loss.numel()} tokens, "
            f"but router_logits has {router_logits.shape[0]} tokens."
        )

    if task_valid_mask is not None and task_valid_mask.numel() == 1:
        task_valid_mask = task_valid_mask.expand_as(task_loss_per_token)

    if task_valid_mask is None:
        task_valid_mask = torch.ones_like(task_loss_per_token, dtype=torch.bool)
    elif task_valid_mask.shape != task_loss_per_token.shape:
        raise ValueError(
            f"task_valid_mask has shape {tuple(task_valid_mask.shape)}, "
            f"but task_loss_per_token has shape {tuple(task_loss_per_token.shape)}."
        )

    if attention_mask is not None and attention_mask.shape != task_valid_mask.shape:
        raise ValueError(
            f"attention_mask has shape {tuple(attention_mask.shape)}, "
            f"but task_valid_mask has shape {tuple(task_valid_mask.shape)}."
        )

    task_valid_mask = task_valid_mask.to(
        device=router_logits.device,
        dtype=torch.bool,
    )

    combined_mask = task_valid_mask.reshape(-1)
    if attention_mask is not None:
        attention_valid_mask = _flatten_attention_mask(attention_mask, router_logits.device)
        if attention_valid_mask.numel() != router_logits.shape[0]:
            raise ValueError(
                f"attention_mask has {attention_valid_mask.numel()} tokens, "
                f"but router_logits has {router_logits.shape[0]} tokens."
            )
        combined_mask = combined_mask & attention_valid_mask

    router_logits = router_logits[combined_mask]
    task_loss = task_loss[combined_mask]

    if router_logits.numel() == 0:
        return router_logits.new_zeros(())

    e = torch.relu(router_logits.float())
    all_evidence = e.sum(dim=-1)
    total_strength = float(num_experts) + all_evidence

    topk_idx = torch.topk(e.detach(), k=top_k, dim=-1).indices
    top_mask = torch.zeros_like(e)
    top_mask.scatter_(-1, topk_idx, 1.0)
    top_evidence = (e * top_mask).sum(dim=-1)
    tail_evidence = all_evidence - top_evidence

    uncertainty = float(num_experts) / (float(num_experts) + all_evidence + eps)
    rho = _truncated_exploration_coeff(uncertainty, u_threshold, eps).detach()
    seek_strength = float(num_experts) + top_evidence + rho * tail_evidence

    clipped_task_loss = task_loss.detach().float().clamp(min=loss_min, max=loss_max)
    loss_per_token = (
        clipped_task_loss * torch.log(total_strength + eps)
        + eta * float(num_experts) / (seek_strength + eps)
    )

    return coef * loss_per_token.mean()
