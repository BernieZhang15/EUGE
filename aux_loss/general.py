from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .loss_utils import (
    _apply_attention_mask,
    _apply_attention_mask_to_expert_outputs,
    _flatten_attention_mask,
    _flatten_router_logits,
    _truncated_exploration_coeff,
    build_euge_intermediates,
)


def _apply_flattened_selection(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    if mask.numel() != tensor.shape[0]:
        raise ValueError(
            f"mask has {mask.numel()} entries, but tensor has {tensor.shape[0]} token rows."
        )
    return tensor[mask]


def topk_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    flat_attention_mask: Optional[torch.Tensor] = None,
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
    if flat_attention_mask is not None:
        router_logits = _apply_flattened_selection(router_logits, flat_attention_mask)
    else:
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
    routing_weights: Optional[torch.Tensor] = None,
    router_scores: Optional[torch.Tensor] = None,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    flat_attention_mask: Optional[torch.Tensor] = None,
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
    source = router_scores if router_scores is not None else routing_weights
    if source is None:
        raise ValueError("discriminative_routing_loss requires routing_weights or router_scores.")
    if coef == 0.0:
        return source.new_zeros(())

    num_experts = source.shape[-1]
    source = _flatten_router_logits(source, num_experts)
    if flat_attention_mask is not None:
        source = _apply_flattened_selection(source, flat_attention_mask)
    else:
        source = _apply_attention_mask(source, attention_mask)

    if source.numel() == 0:
        return source.new_zeros(())

    source = source.float()
    variance = torch.var(source, dim=-1).mean()
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
    active_counts = active_mask.sum(dim=-1)

    # Fast path for the common top-k=2 case. This is algebraically identical
    # to the repo's Gram-Schmidt procedure because each expert is projected
    # onto the 1D subspace spanned by the other expert.
    if expert_outputs.shape[1] == 2 and torch.all(active_counts == 2):
        first = expert_outputs[:, 0, :]
        second = expert_outputs[:, 1, :]
        dot = (first * second).sum(dim=-1)
        first_norm_sq = first.square().sum(dim=-1) + 1e-6
        second_norm_sq = second.square().sum(dim=-1) + 1e-6
        per_token_loss = 0.5 * dot.square() * (
            first_norm_sq.reciprocal() + second_norm_sq.reciprocal()
        )
        return coef * per_token_loss.mean()

    total_loss = expert_outputs.new_zeros(())
    total_count = 0
    for token_idx in range(expert_outputs.shape[0]):
        if active_counts[token_idx].item() <= 1:
            continue
        token_active = active_mask[token_idx]
        active_indices = torch.nonzero(token_active, as_tuple=False).flatten()
        token_outputs = expert_outputs[token_idx, active_indices]
        num_active = token_outputs.shape[0]
        for current_idx in range(num_active):
            other_experts = torch.cat(
                (token_outputs[:current_idx], token_outputs[current_idx + 1 :]),
                dim=0,
            )
            basis = []
            for other_idx in range(other_experts.shape[0]):
                vec = other_experts[other_idx]
                for basis_vec in basis:
                    vec = vec - (basis_vec * vec).sum() * basis_vec
                vec_norm = torch.norm(vec)
                if vec_norm.item() > 1e-6:
                    basis.append(vec / (vec_norm + 1e-6))

            current_expert = token_outputs[current_idx]
            proj_loss = expert_outputs.new_zeros(())
            for basis_vec in basis:
                proj = (basis_vec * current_expert).sum() * basis_vec
                proj_loss = proj_loss + torch.sum(proj.square())
            total_loss = total_loss + proj_loss
            total_count += 1

    if total_count == 0:
        return expert_outputs.new_zeros(())
    return coef * (total_loss / float(total_count))


def evidential_sparsity_loss(
    router_logits: torch.Tensor,
    top_k: int,
    u_threshold: float,
    lambda_sparse: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    evidence: Optional[torch.Tensor] = None,
    selected_experts: Optional[torch.Tensor] = None,
    exploration_coeff: Optional[torch.Tensor] = None,
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

    if (
        evidence is None
        or selected_experts is None
        or exploration_coeff is None
    ):
        intermediates = build_euge_intermediates(
            router_logits=router_logits,
            top_k=top_k,
            num_experts=num_experts,
            u_threshold=u_threshold,
            eps=eps,
        )
        e = intermediates.evidence
        topk_idx = intermediates.selected_experts
        rho = intermediates.exploration_coeff.reshape(-1)
    else:
        e = _apply_attention_mask(evidence, attention_mask).float()
        topk_idx = _apply_attention_mask(selected_experts, attention_mask)
        flat_attention_mask = _flatten_attention_mask(attention_mask, e.device)
        if flat_attention_mask is None:
            rho = exploration_coeff.reshape(-1)
        else:
            rho = _apply_flattened_selection(exploration_coeff, flat_attention_mask)

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
    evidence: Optional[torch.Tensor] = None,
    top_evidence: Optional[torch.Tensor] = None,
    tail_evidence: Optional[torch.Tensor] = None,
    exploration_coeff: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        zero = router_logits.new_zeros(())
        return zero, zero, zero
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
        zero = router_logits.new_zeros(())
        return zero, zero, zero

    if (
        evidence is None
        or top_evidence is None
        or tail_evidence is None
        or exploration_coeff is None
    ):
        intermediates = build_euge_intermediates(
            router_logits=router_logits,
            top_k=top_k,
            num_experts=num_experts,
            u_threshold=u_threshold,
            eps=eps,
        )
        e = intermediates.evidence
        all_evidence = intermediates.evidence_sum.squeeze(-1)
        top_evidence = intermediates.top_evidence
        tail_evidence = intermediates.tail_evidence
        rho = intermediates.exploration_coeff.reshape(-1)
    else:
        e = evidence[combined_mask].float()
        all_evidence = e.sum(dim=-1)
        top_evidence = _apply_flattened_selection(top_evidence, combined_mask)
        tail_evidence = _apply_flattened_selection(tail_evidence, combined_mask)
        rho = _apply_flattened_selection(exploration_coeff.reshape(-1), combined_mask)

    total_strength = float(num_experts) + all_evidence
    seek_strength = float(num_experts) + top_evidence + rho * tail_evidence

    clipped_task_loss = task_loss.detach().float().clamp(min=loss_min, max=loss_max)
    task_fit_term = clipped_task_loss * torch.log(total_strength + eps)
    seek_term = eta * float(num_experts) / (seek_strength + eps)

    task_fit_loss = coef * task_fit_term.mean()
    seek_loss = coef * seek_term.mean()
    total_loss = task_fit_loss + seek_loss

    return total_loss, task_fit_loss, seek_loss
