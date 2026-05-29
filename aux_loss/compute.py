import math
from typing import Any, Dict, Optional, Tuple

import torch

from mixlora.utils import get_mixlora_moe_modules

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


def _round_scale_to_power_of_ten(scale: float, eps: float) -> float:
    if scale <= eps:
        return 0.0
    return float(10 ** round(math.log10(scale)))


def _resolve_repo_style_loss_scale(
    target_value: float,
    raw_loss: torch.Tensor,
    eps: float,
) -> float:
    if target_value == 0.0:
        return 0.0

    raw_magnitude = float(raw_loss.detach().abs().cpu())
    if raw_magnitude <= eps:
        return 0.0

    scale = target_value / raw_magnitude
    scale = _round_scale_to_power_of_ten(scale, eps)
    return scale


def _maybe_init_loss_scale(
    state: Optional[Dict[str, float]],
    scale_key: str,
    target_value: float,
    raw_loss: torch.Tensor,
    eps: float,
) -> None:
    if state is None:
        return
    if scale_key in state:
        return
    if target_value == 0.0:
        state[scale_key] = 0.0
        return
    raw_magnitude = float(raw_loss.detach().abs().cpu())
    if raw_magnitude <= eps:
        return
    state[scale_key] = _resolve_repo_style_loss_scale(
        target_value=target_value,
        raw_loss=raw_loss,
        eps=eps,
    )


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

    total_assignments = raw_stats.get("total_assignments", 0.0)
    token_rows = raw_stats.get("token_rows", 0.0)
    if token_rows > 0.0:
        active_assignments_0 = raw_stats.get("active_assignments_0", 0.0)
        active_assignments_1e_3 = raw_stats.get("active_assignments_1e_3", 0.0)
        expert_sparsity_stats["active_experts_per_token_0"] = active_assignments_0 / token_rows
        expert_sparsity_stats["active_experts_per_token_1e_3"] = (
            active_assignments_1e_3 / token_rows
        )
    if total_assignments > 0.0:
        active_assignments_0 = raw_stats.get("active_assignments_0", 0.0)
        expert_sparsity_stats["routing_sparsity"] = 1.0 - (
            active_assignments_0 / total_assignments
        )

    metric_specs = [
        ("uncertainty", "uncertainty_sum", "uncertainty_count"),
        ("exploration_coeff", "exploration_coeff_sum", "exploration_coeff_count"),
        ("exploration_frac", "exploration_active_sum", "exploration_active_count"),
        ("top_evidence", "top_evidence_sum", "top_evidence_count"),
        ("tail_evidence", "tail_evidence_sum", "tail_evidence_count"),
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
    loss_scale_state: Optional[Dict[str, float]] = None,
    discriminative_target_magnitude: float = 1e-2,
    ortho_target_magnitude: float = 1e-4,
    loss_scale_eps: float = 1e-12,
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
    load_balance_enabled = load_balance_coef > 0.0
    discriminative_enabled = discriminative_coef > 0.0
    evidential_sparsity_enabled = evidential_sparsity_coef > 0.0
    evidence_enabled = evidence_calibration_coef > 0.0
    ortho_enabled = expert_ortho_coef > 0.0
    stats_only_mode = not (
        load_balance_enabled
        or discriminative_enabled
        or evidential_sparsity_enabled
        or evidence_enabled
        or ortho_enabled
    )

    load_balance_losses = []
    discriminative_losses = []
    evidential_sparsity_losses = []
    evidence_losses = []
    evidence_task_fit_losses = []
    evidence_seek_losses = []
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
            active_mask_0 = _apply_flat_mask(
                routing_weights > 0,
                flat_attention_mask,
                "routing_weights",
            )
            active_mask_1e_3 = _apply_flat_mask(
                routing_weights > 1e-3,
                flat_attention_mask,
                "routing_weights",
            )
            if active_mask_0.numel() > 0:
                router_stats_raw["active_assignments_0"] += active_mask_0.float().sum().to(
                    dtype=torch.float64
                )
                router_stats_raw["active_assignments_1e_3"] += active_mask_1e_3.float().sum().to(
                    dtype=torch.float64
                )
                router_stats_raw["total_assignments"] += torch.tensor(
                    active_mask_0.numel(),
                    device=device,
                    dtype=torch.float64,
                )
                router_stats_raw["token_rows"] += torch.tensor(
                    active_mask_0.shape[0],
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

            top_evidence_values = runtime.top_evidence
            tail_evidence_values = runtime.tail_evidence
            if top_evidence_values is not None and tail_evidence_values is not None:
                top_evidence = _apply_flat_mask(
                    top_evidence_values.float().reshape(-1),
                    flat_attention_mask,
                    "top_evidence",
                )
                tail_evidence = _apply_flat_mask(
                    tail_evidence_values.float().reshape(-1),
                    flat_attention_mask,
                    "tail_evidence",
                )

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

        if stats_only_mode:
            continue

        if ortho_enabled:
            ortho_loss = runtime.ortho_loss
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)

        if load_balance_enabled:
            load_balance_losses.append(
                topk_load_balancing_loss(
                    router_logits=router_logits,
                    num_experts=num_experts,
                    top_k=top_k,
                    coef=1.0,
                    flat_attention_mask=flat_attention_mask,
                )
            )

        if discriminative_enabled:
            if routing_weights is None:
                raise RuntimeError(
                    "discriminative_loss requires routing_weights to be available "
                    "in the MoE runtime state."
                )
            dense_router_scores = None
            if routing_strategy == "top-k":
                dense_router_scores = (
                    router_probs.float()
                    if router_probs is not None
                    else torch.softmax(router_logits.float(), dim=-1)
                )
            discriminative_losses.append(
                discriminative_routing_loss(
                    routing_weights=routing_weights,
                    router_scores=dense_router_scores,
                    coef=1.0,
                    flat_attention_mask=flat_attention_mask,
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
                    evidence=getattr(runtime, "evidence", None),
                    selected_experts=getattr(runtime, "selected_experts", None),
                    exploration_coeff=getattr(runtime, "exploration_coeff", None),
                )
            )

        if evidence_enabled:
            (
                evidence_loss,
                evidence_task_fit_loss,
                evidence_seek_loss,
            ) = evidence_calibration_loss(
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
                evidence=getattr(runtime, "evidence", None),
                top_evidence=getattr(runtime, "top_evidence", None),
                tail_evidence=getattr(runtime, "tail_evidence", None),
                exploration_coeff=getattr(runtime, "exploration_coeff", None),
            )
            evidence_losses.append(evidence_loss)
            evidence_task_fit_losses.append(evidence_task_fit_loss)
            evidence_seek_losses.append(evidence_seek_loss)

    ortho_loss_raw = _mean_or_zero(ortho_losses, device)
    load_balance_loss_raw = _mean_or_zero(load_balance_losses, device)
    discriminative_loss_raw = _mean_or_zero(discriminative_losses, device)
    evidential_sparsity_loss_value = _mean_or_zero(evidential_sparsity_losses, device)
    evidence_calibration = _mean_or_zero(evidence_losses, device)
    evidence_calibration_task_fit = _mean_or_zero(evidence_task_fit_losses, device)
    evidence_calibration_seek = _mean_or_zero(evidence_seek_losses, device)

    load_balance_loss = load_balance_coef * load_balance_loss_raw
    discriminative_internal_scale = 1.0
    ortho_internal_scale = 1.0
    discriminative_loss_aligned = discriminative_loss_raw
    ortho_loss_aligned = ortho_loss_raw

    if loss_scale_state is not None:
        if discriminative_coef > 0.0:
            _maybe_init_loss_scale(
                state=loss_scale_state,
                scale_key="discriminative_internal_scale",
                target_value=discriminative_target_magnitude,
                raw_loss=discriminative_loss_raw,
                eps=loss_scale_eps,
            )
        if expert_ortho_coef > 0.0:
            _maybe_init_loss_scale(
                state=loss_scale_state,
                scale_key="ortho_internal_scale",
                target_value=ortho_target_magnitude,
                raw_loss=ortho_loss_raw,
                eps=loss_scale_eps,
            )

    use_loss_scale = (
        loss_scale_state is not None
        and (discriminative_coef > 0.0 or expert_ortho_coef > 0.0)
    )
    if use_loss_scale:
        discriminative_internal_scale = float(
            loss_scale_state.get("discriminative_internal_scale", 1.0)
            if discriminative_coef > 0.0
            else 1.0
        )
        ortho_internal_scale = float(
            loss_scale_state.get("ortho_internal_scale", 1.0)
            if expert_ortho_coef > 0.0
            else 1.0
        )
        discriminative_loss_aligned = discriminative_loss_raw * discriminative_internal_scale
        ortho_loss_aligned = ortho_loss_raw * ortho_internal_scale

    discriminative_loss_value = discriminative_coef * discriminative_loss_aligned
    ortho_loss = expert_ortho_coef * ortho_loss_aligned
    aux_loss = (
        load_balance_loss
        + discriminative_loss_value
        + evidential_sparsity_loss_value
        + evidence_calibration
        + ortho_loss
    )

    stats = {
        "aux_loss": float(aux_loss.detach().cpu()),
        "load_balance_loss": float(load_balance_loss.detach().cpu()),
        "load_balance_loss_raw": float(load_balance_loss_raw.detach().cpu()),
        "discriminative_loss": float(discriminative_loss_value.detach().cpu()),
        "discriminative_loss_raw": float(discriminative_loss_raw.detach().cpu()),
        "discriminative_loss_aligned": float(discriminative_loss_aligned.detach().cpu()),
        "discriminative_internal_scale": float(discriminative_internal_scale),
        "discriminative_coef": float(discriminative_coef),
        "discriminative_target_magnitude": float(discriminative_target_magnitude),
        "evidential_sparsity_loss": float(
            evidential_sparsity_loss_value.detach().cpu()
        ),
        "evidence_calibration_loss": float(evidence_calibration.detach().cpu()),
        "evidence_calibration_task_fit_loss": float(
            evidence_calibration_task_fit.detach().cpu()
        ),
        "evidence_calibration_seek_loss": float(
            evidence_calibration_seek.detach().cpu()
        ),
        "ortho_loss": float(ortho_loss.detach().cpu()),
        "ortho_loss_raw": float(ortho_loss_raw.detach().cpu()),
        "ortho_loss_aligned": float(ortho_loss_aligned.detach().cpu()),
        "ortho_internal_scale": float(ortho_internal_scale),
        "expert_ortho_coef": float(expert_ortho_coef),
        "ortho_target_magnitude": float(ortho_target_magnitude),
        "num_moe_layers": num_moe_layers,
    }

    return aux_loss, stats, _finalize_router_stats(router_stats_raw)
