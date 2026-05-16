from dataclasses import dataclass
from typing import Optional

import torch

from .routing import RoutingResult


@dataclass
class MixLoraRuntimeState:
    router_logits: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None
    router_probs: Optional[torch.Tensor] = None
    tsallis_entropy: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    exploration_coeff: Optional[torch.Tensor] = None
    exploration_mask: Optional[torch.Tensor] = None
    selected_experts: Optional[torch.Tensor] = None
    ortho_loss: Optional[torch.Tensor] = None

    def reset_for_forward(self) -> None:
        self.ortho_loss = None

    def clear_routing_stats(self) -> None:
        self.routing_weights = None
        self.router_probs = None
        self.tsallis_entropy = None
        self.uncertainty = None
        self.exploration_coeff = None
        self.exploration_mask = None
        self.selected_experts = None

    def apply_routing_result(self, routing_result: RoutingResult) -> torch.Tensor:
        self.clear_routing_stats()
        self.routing_weights = routing_result.routing_weights
        self.router_probs = routing_result.router_probs
        self.tsallis_entropy = routing_result.tsallis_entropy
        self.uncertainty = routing_result.uncertainty
        self.exploration_coeff = routing_result.exploration_coeff
        self.exploration_mask = routing_result.exploration_mask
        self.selected_experts = routing_result.selected_experts
        return routing_result.routing_weights
