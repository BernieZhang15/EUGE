from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .loss_utils import _apply_attention_mask, _flatten_router_logits


DYNMOLE_LOSS_TOP_K = 2
DYNMOLE_LOAD_BALANCE_LOSS_COEF = 1e-3
DYNMOLE_ENTROPY_INDEX = 1.1
DYNMOLE_ENTROPY_LOSS_COEF = 1e-2


def dynmole_load_balancing_loss(
    router_probs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DynMoLE uses the Fedus/Switch-style router-balancing term on the router
    probability distribution itself, with fixed paper defaults.
    """
    num_experts = router_probs.shape[-1]
    router_probs = _flatten_router_logits(router_probs, num_experts)
    router_probs = _apply_attention_mask(router_probs, attention_mask)

    if router_probs.numel() == 0:
        return router_probs.new_zeros(())

    router_probs = router_probs.float()
    _, selected = torch.topk(router_probs, DYNMOLE_LOSS_TOP_K, dim=-1)
    expert_mask = F.one_hot(selected, num_classes=num_experts).float()
    tokens_per_expert = expert_mask.sum(dim=(0, 1))
    expert_fraction = tokens_per_expert / tokens_per_expert.sum().clamp(min=1e-6)
    router_prob_per_expert = router_probs.mean(dim=0)
    loss = num_experts * torch.sum(expert_fraction * router_prob_per_expert)
    return DYNMOLE_LOAD_BALANCE_LOSS_COEF * loss


def tsallis_entropy_loss(
    router_probs: torch.Tensor,
    num_experts: int,
    q: float,
    coef: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if coef == 0.0:
        return router_probs.new_zeros(())

    router_probs = _flatten_router_logits(router_probs, num_experts)
    router_probs = _apply_attention_mask(router_probs, attention_mask)

    if router_probs.numel() == 0:
        return router_probs.new_zeros(())

    router_probs = router_probs.float()
    if abs(q - 1.0) < 1e-6:
        entropy = -(router_probs * router_probs.clamp(min=1e-12).log()).sum(dim=-1)
    else:
        entropy = (1.0 - router_probs.pow(q).sum(dim=-1)) / (q - 1.0)
    return coef * entropy.mean()


def dynmole_entropy_loss(
    router_probs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_experts = router_probs.shape[-1]
    return tsallis_entropy_loss(
        router_probs=router_probs,
        num_experts=num_experts,
        q=DYNMOLE_ENTROPY_INDEX,
        coef=DYNMOLE_ENTROPY_LOSS_COEF,
        attention_mask=attention_mask,
    )


def dynmole_auxiliary_loss(
    router_probs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper-aligned DynMoLE auxiliary terms with fixed defaults:
      - load balance loss: Fedus/Switch-style, top-k=2, coef=1e-3
      - Tsallis entropy loss: q=1.1, coef=1e-2
    """
    load_balance = dynmole_load_balancing_loss(
        router_probs=router_probs,
        attention_mask=attention_mask,
    )
    entropy = dynmole_entropy_loss(
        router_probs=router_probs,
        attention_mask=attention_mask,
    )
    return load_balance, entropy
