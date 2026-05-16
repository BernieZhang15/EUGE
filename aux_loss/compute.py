from typing import Any, Dict, Optional, Tuple

import torch

from mixlora.utils import get_mixlora_moe_modules

from .dynmole import DYNMOLE_ENTROPY_LOSS_COEF, dynmole_auxiliary_loss
from .general import (
    discriminative_routing_loss,
    evidence_calibration_loss,
    evidential_sparsity_loss,
    topk_load_balancing_loss,
)
from .loss_utils import (
    _apply_flat_mask,
    _empty_router_stats_raw,
    _flatten_attention_mask,
    _mean_or_zero,
)
from .remoe import remoe_regularization_loss


def _init_router_stats_accumulator(device) -> Dict[str, torch.Tensor]:
    return {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in _empty_router_stats_raw()
    }


def _finalize_router_stats(
    router_stats_raw: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        key: float(value.item())
        for key, value in router_stats_raw.items()
    }


def summarize_router_stats(
    raw_stats: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    euge_stats: Dict[str, float] = {}
    expert_sparsity_stats: Dict[str, float] = {}

    active_assignments = raw_stats.get("active_assignments", 0.0)
    total_assignments = raw_stats.get("total_assignments", 0.0)
    token_rows = raw_stats.get("token_rows", 0.0)
    if token_rows > 0.0:
        expert_sparsity_stats["active_experts_per_token"] = active_assignments / token_rows
    if total_assignments > 0.0:
        expert_sparsity_stats["routing_sparsity"] = 1.0 - (
            active_assignments / total_assignments
        )

    metric_specs = [
        ("uncertainty", "uncertainty_sum", "uncertainty_count"),
        ("exploration_coeff", "exploration_coeff_sum", "exploration_coeff_count"),
        ("exploration_frac", "exploration_active_sum", "exploration_active_count"),
        ("top_evidence", "top_evidence_sum", "top_evidence_count"),
        ("tail_evidence", "tail_evidence_sum", "tail_evidence_count"),
        ("tail_ratio", "tail_ratio_sum", "tail_ratio_count"),
    ]
    for output_key, sum_key, count_key in metric_specs:
        count = raw_stats.get(count_key, 0.0)
        if count > 0.0:
            euge_stats[output_key] = raw_stats.get(sum_key, 0.0) / count

    return euge_stats, expert_sparsity_stats


def compute_aux_loss(
    model: torch.nn.Module,
    routing_strategy: str,
    num_experts: int,
    top_k: Optional[int],
    load_balance_coef: float,
    discriminative_coef: float,
    evidential_sparsity_coef: float,
    expert_ortho_coef: float,
    remoe_reg_coef: float,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sparsity_eps: float = 1e-8,
    u_threshold: float = 0.1,
    evidence_calibration_coef: float = 0.0,
    evidence_eta: float = 1.0,
    evidence_loss_min: float = 0.0,
    evidence_loss_max: float = 3.0,
    task_valid_mask: Optional[torch.Tensor] = None,
    task_loss_per_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, float]]:
    """
    Compute auxiliary loss.

    Any auxiliary loss with a non-zero coefficient is added, regardless of
    routing strategy:

        aux_loss =
            mean_l L_load_balance^{(l)}
          + mean_l L_discriminative^{(l)}
          + mean_l L_evidential_sparse^{(l)}
          + mean_l L_calibration^{(l)}
          + mean_l L_ortho^{(l)}
    """
    dynmole_fixed_loss_enabled = routing_strategy == "dynmole"
    load_balance_enabled = load_balance_coef > 0.0 or dynmole_fixed_loss_enabled
    dynmole_entropy_enabled = dynmole_fixed_loss_enabled
    remoe_reg_enabled = (
        routing_strategy == "remoe"
        and remoe_reg_coef > 0.0
        and not dynmole_fixed_loss_enabled
    )
    discriminative_enabled = discriminative_coef > 0.0 and not dynmole_fixed_loss_enabled
    evidential_sparsity_enabled = (
        evidential_sparsity_coef > 0.0 and not dynmole_fixed_loss_enabled
    )
    evidence_enabled = evidence_calibration_coef > 0.0 and not dynmole_fixed_loss_enabled
    ortho_enabled = (
        expert_ortho_coef > 0.0
        and routing_strategy != "remoe"
        and not dynmole_fixed_loss_enabled
    )
    stats_only_mode = not (
        load_balance_enabled
        or dynmole_entropy_enabled
        or remoe_reg_enabled
        or discriminative_enabled
        or evidential_sparsity_enabled
        or evidence_enabled
        or ortho_enabled
    )

    load_balance_losses = []
    dynmole_entropy_losses = []
    remoe_reg_losses = []
    discriminative_losses = []
    evidential_sparsity_losses = []
    evidence_losses = []
    ortho_losses = []
    num_moe_layers = 0
    router_stats_raw = _init_router_stats_accumulator(device)
    flat_attention_mask = _flatten_attention_mask(attention_mask, device)

    for module in get_mixlora_moe_modules(model):
        runtime = getattr(module, "runtime_", None)
        if runtime is None:
            continue

        router_logits = runtime.router_logits

        if router_logits is None:
            continue

        routing_weights = runtime.routing_weights
        router_probs = runtime.router_probs

        num_moe_layers += 1

        if routing_weights is not None:
            active_mask = _apply_flat_mask(
                routing_weights > 0,
                flat_attention_mask,
                "routing_weights",
            )
            if active_mask.numel() > 0:
                router_stats_raw["active_assignments"] += active_mask.float().sum().to(
                    dtype=torch.float64
                )
                router_stats_raw["total_assignments"] += torch.tensor(
                    active_mask.numel(),
                    device=device,
                    dtype=torch.float64,
                )
                router_stats_raw["token_rows"] += torch.tensor(
                    active_mask.shape[0],
                    device=device,
                    dtype=torch.float64,
                )

        if routing_strategy == "EUGE":
            uncertainty = runtime.uncertainty
            if uncertainty is not None:
                uncertainty_values = _apply_flat_mask(
                    uncertainty.float().reshape(-1),
                    flat_attention_mask,
                    "uncertainty",
                )
                if uncertainty_values.numel() > 0:
                    router_stats_raw["uncertainty_sum"] += uncertainty_values.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["uncertainty_count"] += torch.tensor(
                        uncertainty_values.numel(),
                        device=device,
                        dtype=torch.float64,
                    )

            exploration_coeff = runtime.exploration_coeff
            if exploration_coeff is not None:
                exploration_coeff_values = _apply_flat_mask(
                    exploration_coeff.float().reshape(-1),
                    flat_attention_mask,
                    "exploration_coeff",
                )
                if exploration_coeff_values.numel() > 0:
                    router_stats_raw["exploration_coeff_sum"] += exploration_coeff_values.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["exploration_coeff_count"] += torch.tensor(
                        exploration_coeff_values.numel(),
                        device=device,
                        dtype=torch.float64,
                    )

            exploration_mask = runtime.exploration_mask
            if exploration_mask is not None:
                exploration_values = _apply_flat_mask(
                    exploration_mask.float().reshape(-1),
                    flat_attention_mask,
                    "exploration_mask",
                )
                if exploration_values.numel() > 0:
                    router_stats_raw["exploration_active_sum"] += exploration_values.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["exploration_active_count"] += torch.tensor(
                        exploration_values.numel(),
                        device=device,
                        dtype=torch.float64,
                    )

            selected_experts = runtime.selected_experts
            if router_logits is not None and selected_experts is not None:
                evidence = torch.relu(router_logits.float())
                selected_experts = selected_experts.to(device=evidence.device)
                top_mask = torch.zeros_like(evidence, dtype=torch.bool)
                top_mask.scatter_(dim=-1, index=selected_experts, value=True)

                top_evidence = _apply_flat_mask(
                    (evidence * top_mask.float()).sum(dim=-1),
                    flat_attention_mask,
                    "top_evidence",
                )
                tail_evidence = _apply_flat_mask(
                    (evidence * (~top_mask).float()).sum(dim=-1),
                    flat_attention_mask,
                    "tail_evidence",
                )
                tail_ratio = tail_evidence / (top_evidence + tail_evidence + 1e-8)

                if top_evidence.numel() > 0:
                    router_stats_raw["top_evidence_sum"] += top_evidence.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["top_evidence_count"] += torch.tensor(
                        top_evidence.numel(),
                        device=device,
                        dtype=torch.float64,
                    )
                    router_stats_raw["tail_evidence_sum"] += tail_evidence.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["tail_evidence_count"] += torch.tensor(
                        tail_evidence.numel(),
                        device=device,
                        dtype=torch.float64,
                    )
                    router_stats_raw["tail_ratio_sum"] += tail_ratio.sum().to(
                        dtype=torch.float64
                    )
                    router_stats_raw["tail_ratio_count"] += torch.tensor(
                        tail_ratio.numel(),
                        device=device,
                        dtype=torch.float64,
                    )

        if stats_only_mode:
            continue

        if ortho_enabled:
            ortho_loss = runtime.ortho_loss
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)

        if load_balance_enabled:
            if routing_strategy == "dynmole":
                if router_probs is None:
                    raise RuntimeError(
                        "DynMoLE auxiliary loss requires router_probs "
                        "to be available in the MoE runtime state."
                    )
                dynmole_load_balance, dynmole_entropy = dynmole_auxiliary_loss(
                    router_probs=router_probs,
                    attention_mask=attention_mask,
                )
                load_balance_losses.append(dynmole_load_balance)
                dynmole_entropy_losses.append(dynmole_entropy)
            else:
                load_balance_losses.append(
                    topk_load_balancing_loss(
                        router_logits=router_logits,
                        num_experts=num_experts,
                        top_k=top_k,
                        coef=load_balance_coef,
                        attention_mask=attention_mask,
                    )
                )

        if remoe_reg_enabled:
            if routing_weights is None:
                raise RuntimeError(
                    "remoe_regularization_loss requires routing_weights "
                    "to be available in the MoE runtime state."
                )
            remoe_reg_losses.append(
                remoe_regularization_loss(
                    routing_weights=routing_weights,
                    num_experts=num_experts,
                    top_k=top_k,
                    coef=remoe_reg_coef,
                    attention_mask=attention_mask,
                )
            )

        if discriminative_enabled:
            if routing_weights is None:
                raise RuntimeError(
                    "discriminative_loss requires routing_weights to be available "
                    "in the MoE runtime state."
                )
            discriminative_losses.append(
                discriminative_routing_loss(
                    routing_weights=routing_weights,
                    coef=discriminative_coef,
                    attention_mask=attention_mask,
                )
            )

        if evidential_sparsity_enabled:
            evidential_sparsity_losses.append(
                evidential_sparsity_loss(
                    router_logits=router_logits,
                    top_k=top_k,
                    u_threshold=u_threshold,
                    lambda_sparse=evidential_sparsity_coef,
                    attention_mask=attention_mask,
                    eps=sparsity_eps,
                )
            )

        if evidence_enabled:
            evidence_loss = evidence_calibration_loss(
                router_logits=router_logits,
                top_k=top_k,
                u_threshold=u_threshold,
                coef=evidence_calibration_coef,
                eta=evidence_eta,
                eps=sparsity_eps,
                attention_mask=attention_mask,
                task_valid_mask=task_valid_mask,
                task_loss_per_token=task_loss_per_token,
                loss_min=evidence_loss_min,
                loss_max=evidence_loss_max,
            )
            evidence_losses.append(evidence_loss)

    ortho_loss = _mean_or_zero(ortho_losses, device)
    load_balance_loss = _mean_or_zero(load_balance_losses, device)
    dynmole_entropy_loss_value = _mean_or_zero(dynmole_entropy_losses, device)
    remoe_reg_loss = _mean_or_zero(remoe_reg_losses, device)
    discriminative_loss_value = _mean_or_zero(discriminative_losses, device)
    evidential_sparsity_loss_value = _mean_or_zero(evidential_sparsity_losses, device)
    evidence_calibration = _mean_or_zero(evidence_losses, device)
    aux_loss = (
        load_balance_loss
        + dynmole_entropy_loss_value
        + remoe_reg_loss
        + discriminative_loss_value
        + evidential_sparsity_loss_value
        + evidence_calibration
        + ortho_loss
    )

    stats = {
        "aux_loss": float(aux_loss.detach().cpu()),
        "load_balance_loss": float(load_balance_loss.detach().cpu()),
        "dynmole_entropy_loss": float(dynmole_entropy_loss_value.detach().cpu()),
        "remoe_reg_loss": float(remoe_reg_loss.detach().cpu()),
        "discriminative_loss": float(discriminative_loss_value.detach().cpu()),
        "evidential_sparsity_loss": float(
            evidential_sparsity_loss_value.detach().cpu()
        ),
        "evidence_calibration_loss": float(evidence_calibration.detach().cpu()),
        "ortho_loss": float(ortho_loss.detach().cpu()),
        "num_moe_layers": num_moe_layers,
    }
    if remoe_reg_enabled:
        stats["remoe_reg_coef"] = float(remoe_reg_coef)
    if dynmole_entropy_enabled:
        stats["dynmole_entropy_coef"] = float(DYNMOLE_ENTROPY_LOSS_COEF)

    return aux_loss, stats, _finalize_router_stats(router_stats_raw)
