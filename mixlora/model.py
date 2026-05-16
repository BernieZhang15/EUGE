import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from aux_loss import compute_moe_ortho_loss
from .config import MixLoraConfig
from .forward import resolve_expert_forward
from .lora_linear import LoraLinear
from .routing import RoutingResult, compute_routing
from .runtime import MixLoraRuntimeState
from .utils import infer_device

if TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = Any


def _expert_key(expert_idx: int, proj_name: str) -> str:
    """ModuleDict keys must not contain '.', use '_' as separator."""
    return f"expert_{expert_idx}_{proj_name}"


def _get_act_fn(base_layer: nn.Module):
    """Robustly retrieve activation function from a base FFN layer."""
    for attr in ("act_fn", "activation_fn", "hidden_act"):
        fn = getattr(base_layer, attr, None)
        if fn is not None and callable(fn):
            return fn
    raise AttributeError(
        f"Cannot find activation function on {type(base_layer).__name__}. "
        f"Tried: act_fn, activation_fn, hidden_act."
    )


def _gate_key(layer_idx: int) -> str:
    return f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"


def _gate_bias_key(layer_idx: int) -> str:
    return f"mixlora.layers.{layer_idx}.mlp.moe_gate.bias"


def _lora_key(layer_idx: int, proj_name: str, expert_idx: int, ab: str) -> str:
    return f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}.{ab}.weight"


class MixLoraSparseMoe(nn.Module):
    def __init__(self, base_layer: nn.Module, config: MixLoraConfig) -> None:
        super().__init__()

        self._is_mixlora_moe = True
        self.dtype_: torch.dtype = config.dtype_
        self.gate_: nn.Parameter = None
        self.gate_bias_: Optional[nn.Parameter] = None
        object.__setattr__(self, "base_layer_", base_layer)
        self.experts_: nn.ModuleDict = nn.ModuleDict()
        self.expert_groups_: Dict[str, List[Optional[LoraLinear]]] = {}
        self.act_fn_ = _get_act_fn(base_layer)
        self.num_experts_: int = config.num_experts_
        self.topk_: Optional[int] = config.top_k_
        self.routing_strategy_: str = config.routing_strategy_
        self.u_threshold_: float = config.u_threshold_
        self.dynmole_top_p_: float = config.dynmole_top_p_
        self.dynmole_entropy_threshold_: float = config.dynmole_entropy_threshold_
        self.dynmole_entropy_index_: float = config.dynmole_entropy_index_
        self.expert_ortho_loss_coef_: float = config.expert_ortho_loss_coef_
        self.runtime_ = MixLoraRuntimeState()
        self.forward_fn_ = resolve_expert_forward(config.model_type_)

    def _get_expert(self, expert_idx: int, proj_name: str) -> Optional[LoraLinear]:
        key = _expert_key(expert_idx, proj_name)
        return self.experts_[key] if key in self.experts_ else None

    def _get_expert_group(self, proj_name: str) -> List[Optional[LoraLinear]]:
        group = self.expert_groups_.get(proj_name)
        if group is not None:
            return group
        group = [self._get_expert(expert_idx, proj_name) for expert_idx in range(self.num_experts_)]
        self.expert_groups_[proj_name] = group
        return group

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_.to(device=hidden_states.device, dtype=hidden_states.dtype)
        gate_bias = None
        if self.gate_bias_ is not None:
            gate_bias = self.gate_bias_.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        router_logits = F.linear(hidden_states, gate, gate_bias)
        self.runtime_.router_logits = router_logits
        return router_logits

    def _apply_routing_result(self, routing_result: RoutingResult) -> torch.Tensor:
        return self.runtime_.apply_routing_result(routing_result)

    def _build_expert_token_indices(
        self,
        active_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        if active_mask.ndim != 2 or active_mask.shape[-1] != self.num_experts_:
            raise ValueError(
                "active_mask must have shape [num_tokens, num_experts], "
                f"got {tuple(active_mask.shape)}"
            )
        return [
            torch.where(active_mask[:, expert_idx])[0]
            for expert_idx in range(self.num_experts_)
        ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, hidden_dim).to(self.dtype_)
        self.runtime_.reset_for_forward()

        router_logits = self._compute_router_logits(hidden_states)
        routing_weights = self._apply_routing_result(
            compute_routing(
                router_logits=router_logits,
                strategy=self.routing_strategy_,
                num_experts=self.num_experts_,
                top_k=self.topk_,
                dtype=self.dtype_,
                u_threshold=self.u_threshold_,
                dynmole_top_p=self.dynmole_top_p_,
                dynmole_entropy_threshold=self.dynmole_entropy_threshold_,
                dynmole_entropy_index=self.dynmole_entropy_index_,
            )
        )

        expert_token_indices = self._build_expert_token_indices(routing_weights > 0)
        expert_states = self.forward_fn_(
            base_layer=self.base_layer_,
            get_expert_group=self._get_expert_group,
            num_experts=self.num_experts_,
            act_fn=self.act_fn_,
            dtype=self.dtype_,
            expert_token_indices=expert_token_indices,
            hidden_states=hidden_states,
            input_dtype=input_dtype,
        )
        if self.expert_ortho_loss_coef_ > 0.0:
            self.runtime_.ortho_loss = compute_moe_ortho_loss(
                training=self.training,
                routing_strategy=self.routing_strategy_,
                coef=self.expert_ortho_loss_coef_,
                selected_experts=self.runtime_.selected_experts,
                num_experts=self.num_experts_,
                dtype=self.dtype_,
                expert_token_indices=expert_token_indices,
                routing_weights=routing_weights,
                expert_states=expert_states,
                hidden_states=hidden_states,
            )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_, device=hidden_states.device,
        )
        for expert_idx in range(self.num_experts_):
            top_x = expert_token_indices[expert_idx]
            if top_x.numel() == 0:
                continue
            final_hidden_states.index_add_(
                0, top_x,
                (expert_states[expert_idx] * routing_weights[top_x, expert_idx, None]).to(self.dtype_)
            )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim).to(input_dtype)


def collect_adapter_weights(
    model: "PreTrainedModel",
    config: MixLoraConfig,
) -> Dict[str, torch.Tensor]:
    weights: Dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "mixlora_moes"):
            continue
        if config.adapter_name_ not in mlp.mixlora_moes:
            continue
        moe_layer: MixLoraSparseMoe = mlp.mixlora_moes[config.adapter_name_]
        if moe_layer.gate_ is not None:
            weights[_gate_key(layer_idx)] = moe_layer.gate_.detach().cpu()
        if moe_layer.gate_bias_ is not None:
            weights[_gate_bias_key(layer_idx)] = moe_layer.gate_bias_.detach().cpu()
        for proj_name, inject in config.target_modules_.items():
            if not inject:
                continue
            for expert_idx in range(config.num_experts_):
                lora = moe_layer._get_expert(expert_idx, proj_name)
                if lora is None:
                    continue
                weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_A")] = \
                    lora.lora_A.weight.detach().cpu()
                weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_B")] = \
                    lora.lora_B.weight.detach().cpu()
    return weights


def _inject_mlp_module(
    layer_idx: int,
    mlp: nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
) -> None:
    moe_layer = MixLoraSparseMoe(mlp, config)
    moe_layer.gate_ = nn.Parameter(
        weights[_gate_key(layer_idx)].to(config.dtype_),
        requires_grad=True,
    )
    if _gate_bias_key(layer_idx) in weights:
        moe_layer.gate_bias_ = nn.Parameter(
            weights[_gate_bias_key(layer_idx)].to(config.dtype_),
            requires_grad=True,
        )
    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = nn.ModuleDict()
    mlp.mixlora_moes[config.adapter_name_] = moe_layer
    mlp.forward = moe_layer.forward

    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        base_layer = getattr(mlp, proj_name)
        for expert_idx in range(config.num_experts_):
            moe_layer.experts_[_expert_key(expert_idx, proj_name)] = LoraLinear(
                base_layer,
                config,
                (
                    weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_A")],
                    weights[_lora_key(layer_idx, proj_name, expert_idx, "lora_B")],
                ),
            )
    moe_layer.expert_groups_.clear()


def inject_adapter_in_model(
    model: "PreTrainedModel",
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
) -> None:
    config.model_type_ = model.config.model_type
    model._mixlora_config = config
    for idx, layer in enumerate(model.model.layers):
        _inject_mlp_module(idx, layer.mlp, config, weights)


def load_adapter_weights(
    name_or_path: str,
    adapter_name: str = "default",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[MixLoraConfig, Dict[str, torch.Tensor]]:
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")
    if device is None:
        device = infer_device()

    config_path = os.path.join(name_or_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf8") as fp:
        config = MixLoraConfig.from_config(json.load(fp))
        config.adapter_name_ = adapter_name
        config.dtype_ = dtype
    config.check()

    weights: Dict[str, torch.Tensor] = torch.load(
        os.path.join(name_or_path, "adapter_model.bin"),
        map_location=device,
        weights_only=True,
    )
    return config, weights


_compatible_task_types = ["CAUSAL_LM", "QUESTION_ANS"]


class MixLoraModelForCausalLM:
    @staticmethod
    def from_pretrained(
        name_or_path: str,
        *model_args,
        **kwargs,
    ) -> Tuple["PreTrainedModel", MixLoraConfig]:
        from transformers import AutoModelForCausalLM

        adapter_name = kwargs.pop("adapter_name", "default")
        dtype = kwargs.get("torch_dtype", torch.float32)
        load_device = kwargs.get("device")
        if load_device is None:
            device_map = kwargs.get("device_map")
            if isinstance(device_map, str):
                load_device = device_map

        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name=adapter_name,
            device=load_device,
            dtype=dtype,
        )
        assert config.task_type_ in _compatible_task_types, (
            f"Unsupported task type '{config.task_type_}'. "
            f"Supported: {_compatible_task_types}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_, *model_args, **kwargs
        )
        inject_adapter_in_model(model, config, weights)
        return model, config
