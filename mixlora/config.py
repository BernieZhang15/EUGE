import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    coerced = int(value)
    if float(value) != float(coerced):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return coerced


@dataclass
class AdapterConfig:
    base_model_: str = None
    task_type_: str = None
    peft_type_: str = None
    adapter_name_: str = None
    model_type_: str = None
    dtype_: torch.dtype = None

    @property
    def base_model_name_or_path(self):
        return self.base_model_

    @property
    def adapter_name(self):
        return self.adapter_name_

    def check(self) -> "AdapterConfig":
        assert isinstance(self.base_model_, str)
        assert isinstance(self.task_type_, str)
        assert isinstance(self.peft_type_, str)
        return self

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "AdapterConfig":
        return AdapterConfig(
            base_model_=config["base_model_name_or_path"],
            task_type_=config["task_type"],
            peft_type_=config["peft_type"],
        )

    def export(self) -> Dict[str, Any]:
        return {
            "bias": "none",
            "peft_type": self.peft_type_,
            "task_type": self.task_type_,
            "base_model_name_or_path": self.base_model_,
        }


lora_target_modules = {
    # LLaMA names
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "fc1": False,
    "fc2": False,
    # Phi3 names
    "gate_up_proj": False,
}


default_target_modules_by_model_type = {
    "llama": ["gate_proj", "down_proj", "up_proj"],
    "gemma": ["gate_proj", "down_proj", "up_proj"],
    "gemma2": ["gate_proj", "down_proj", "up_proj"],
    "qwen2": ["gate_proj", "down_proj", "up_proj"],
    "mistral": ["gate_proj", "down_proj", "up_proj"],
    "phi": ["fc1", "fc2"],
    "phi3": ["gate_up_proj", "down_proj"],
}


def default_target_modules(model_type: str) -> Dict[str, bool]:
    if model_type not in default_target_modules_by_model_type:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            f"Supported: {list(default_target_modules_by_model_type)}"
        )

    targets = copy.deepcopy(lora_target_modules)
    for target in default_target_modules_by_model_type[model_type]:
        targets[target] = True
    return targets


def resolve_target_modules(
    model_type: str,
    override: Optional[Any] = None,
) -> Dict[str, bool]:
    if override is None:
        return default_target_modules(model_type)

    if isinstance(override, list):
        targets = copy.deepcopy(lora_target_modules)
        for target in override:
            if target not in targets:
                raise ValueError(f"Unknown target module '{target}'")
            targets[target] = True
        return targets

    if isinstance(override, dict):
        targets = default_target_modules(model_type)
        for target, enabled in override.items():
            if target not in targets:
                raise ValueError(f"Unknown target module '{target}'")
            if not isinstance(enabled, bool):
                raise ValueError(f"target_modules['{target}'] must be bool")
            targets[target] = enabled
        return targets

    raise ValueError("target_modules must be None, a list, or a dict")


@dataclass
class LoraConfig(AdapterConfig):
    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    def check(self) -> "LoraConfig":
        super().check()
        assert isinstance(self.lora_init_, str) and self.lora_init_ in ["original", "gaussian"]
        assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_, float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)
        return self

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = _coerce_int(config["r"], "r")
        lora_config.lora_alpha_ = _coerce_int(config["lora_alpha"], "lora_alpha")
        lora_config.lora_dropout_ = _coerce_float(
            config["lora_dropout"],
            "lora_dropout",
        )
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], list):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")
        return lora_config

    def export(self) -> Dict[str, Any]:
        config = super().export()
        config["lora_init"] = self.lora_init_
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        config["target_modules"] = [t for t, v in self.target_modules_.items() if v]
        return config


available_routing_strategies = ["top-k", "dynmole", "remoe", "EUGE"]


def normalize_routing_strategy(routing_strategy: str) -> str:
    if not isinstance(routing_strategy, str):
        return routing_strategy

    normalized = routing_strategy.strip()
    lowered = normalized.lower()
    if lowered == "euge-moe" or lowered == "euge":
        return "EUGE"
    if lowered == "remoe":
        return "remoe"
    if lowered == "dynmole":
        return "dynmole"
    if lowered in {"mixlora", "top-k", "topk"}:
        return "top-k"
    return normalized


@dataclass
class MixLoraConfig(LoraConfig):
    load_balance_loss_coef_: float = None
    discriminative_loss_coef_: float = 0.0
    evidential_sparsity_loss_coef_: float = None
    evidence_calibration_loss_coef_: float = 0.0
    expert_ortho_loss_coef_: float = 0.0
    router_init_range_: float = None
    router_bias_init_: float = 0.0
    u_threshold_: float = 0.1
    dynmole_top_p_: float = 0.75
    dynmole_entropy_threshold_: float = 0.9
    dynmole_entropy_index_: float = 1.1
    dynmole_entropy_loss_coef_: float = 1e-2
    remoe_reg_init_: float = 1e-8
    remoe_reg_update_mul_: float = 1.2
    remoe_target_sparsity_: Optional[float] = None
    routing_strategy_: str = None
    num_experts_: int = None

    top_k_: int = None

    def check(self) -> "MixLoraConfig":
        super().check()
        self.routing_strategy_ = normalize_routing_strategy(self.routing_strategy_)
        assert isinstance(self.load_balance_loss_coef_, float) and self.load_balance_loss_coef_ >= 0
        assert (
            isinstance(self.discriminative_loss_coef_, float)
            and self.discriminative_loss_coef_ >= 0
        )
        assert isinstance(self.evidential_sparsity_loss_coef_, float) and self.evidential_sparsity_loss_coef_ >= 0
        assert (
            isinstance(self.evidence_calibration_loss_coef_, float)
            and self.evidence_calibration_loss_coef_ >= 0
        )
        assert isinstance(self.expert_ortho_loss_coef_, float) and self.expert_ortho_loss_coef_ >= 0
        assert isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        assert isinstance(self.router_bias_init_, float)
        assert isinstance(self.u_threshold_, float) and 0.0 < self.u_threshold_ < 1.0
        assert (
            isinstance(self.dynmole_top_p_, float)
            and 0.0 < self.dynmole_top_p_ <= 1.0
        )
        assert (
            isinstance(self.dynmole_entropy_threshold_, float)
            and self.dynmole_entropy_threshold_ >= 0.0
        )
        assert (
            isinstance(self.dynmole_entropy_index_, float)
            and self.dynmole_entropy_index_ > 0.0
        )
        assert (
            isinstance(self.dynmole_entropy_loss_coef_, float)
            and self.dynmole_entropy_loss_coef_ >= 0.0
        )
        assert isinstance(self.remoe_reg_init_, float) and self.remoe_reg_init_ >= 0
        assert (
            isinstance(self.remoe_reg_update_mul_, float)
            and self.remoe_reg_update_mul_ > 1.0
        )
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert isinstance(self.top_k_, int) and 0 < self.top_k_ <= self.num_experts_, \
            f"top_k must be in (0, num_experts={self.num_experts_}]"
        if self.remoe_target_sparsity_ is None:
            self.remoe_target_sparsity_ = 1.0 - (self.top_k_ / float(self.num_experts_))
        assert (
            isinstance(self.remoe_target_sparsity_, float)
            and 0.0 <= self.remoe_target_sparsity_ < 1.0
        )
        return self

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "MixLoraConfig":
        lora_config = MixLoraConfig(**LoraConfig.from_config(config).__dict__)
        lora_config.routing_strategy_ = normalize_routing_strategy(
            config.get("routing_strategy", None)
        )
        assert (
            lora_config.peft_type_ == "MIXLORA"
            and lora_config.routing_strategy_ is not None
            and lora_config.routing_strategy_ in available_routing_strategies
        ), f"MixLoraConfig only supports routing strategies: {available_routing_strategies}"
        lora_config.load_balance_loss_coef_ = _coerce_float(
            config.get("load_balance_loss_coef", 0.0),
            "load_balance_loss_coef",
        )
        lora_config.discriminative_loss_coef_ = _coerce_float(
            config.get("discriminative_loss_coef", 0.0),
            "discriminative_loss_coef",
        )
        lora_config.evidential_sparsity_loss_coef_ = _coerce_float(
            config.get("evidential_sparsity_loss_coef", 0.0),
            "evidential_sparsity_loss_coef",
        )
        lora_config.evidence_calibration_loss_coef_ = _coerce_float(
            config.get("evidence_calibration_loss_coef", 0.0),
            "evidence_calibration_loss_coef",
        )
        lora_config.expert_ortho_loss_coef_ = _coerce_float(
            config.get("expert_ortho_loss_coef", 0.0),
            "expert_ortho_loss_coef",
        )
        lora_config.num_experts_ = _coerce_int(config["num_experts"], "num_experts")
        lora_config.router_init_range_ = _coerce_float(
            config.get("router_init_range", 0.02),
            "router_init_range",
        )
        default_router_bias_init = (
            -0.02 if lora_config.routing_strategy_ == "remoe" else 0.0
        )
        lora_config.router_bias_init_ = _coerce_float(
            config.get("router_bias_init", default_router_bias_init),
            "router_bias_init",
        )
        lora_config.u_threshold_ = _coerce_float(
            config.get("u_threshold", 0.1),
            "u_threshold",
        )
        lora_config.dynmole_top_p_ = _coerce_float(
            config.get("dynmole_top_p", 0.75),
            "dynmole_top_p",
        )
        lora_config.dynmole_entropy_threshold_ = _coerce_float(
            config.get("dynmole_entropy_threshold", 0.9),
            "dynmole_entropy_threshold",
        )
        lora_config.dynmole_entropy_index_ = _coerce_float(
            config.get("dynmole_entropy_index", 1.1),
            "dynmole_entropy_index",
        )
        lora_config.dynmole_entropy_loss_coef_ = _coerce_float(
            config.get("dynmole_entropy_loss_coef", 1e-2),
            "dynmole_entropy_loss_coef",
        )
        lora_config.remoe_reg_init_ = _coerce_float(
            config.get("remoe_reg_init", 1e-8),
            "remoe_reg_init",
        )
        lora_config.remoe_reg_update_mul_ = _coerce_float(
            config.get("remoe_reg_update_mul", 1.2),
            "remoe_reg_update_mul",
        )
        remoe_target_sparsity = config.get("remoe_target_sparsity")
        lora_config.remoe_target_sparsity_ = (
            None
            if remoe_target_sparsity is None
            else _coerce_float(remoe_target_sparsity, "remoe_target_sparsity")
        )
        lora_config.top_k_ = _coerce_int(config.get("top_k", 2), "top_k")
        return lora_config

    def export(self) -> Dict[str, Any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        config["top_k"] = self.top_k_
        config["load_balance_loss_coef"] = self.load_balance_loss_coef_
        config["discriminative_loss_coef"] = self.discriminative_loss_coef_
        config["evidential_sparsity_loss_coef"] = self.evidential_sparsity_loss_coef_
        config["evidence_calibration_loss_coef"] = self.evidence_calibration_loss_coef_
        config["expert_ortho_loss_coef"] = self.expert_ortho_loss_coef_
        config["router_init_range"] = self.router_init_range_
        config["router_bias_init"] = self.router_bias_init_
        config["u_threshold"] = self.u_threshold_
        config["dynmole_top_p"] = self.dynmole_top_p_
        config["dynmole_entropy_threshold"] = self.dynmole_entropy_threshold_
        config["dynmole_entropy_index"] = self.dynmole_entropy_index_
        config["dynmole_entropy_loss_coef"] = self.dynmole_entropy_loss_coef_
        config["remoe_reg_init"] = self.remoe_reg_init_
        config["remoe_reg_update_mul"] = self.remoe_reg_update_mul_
        config["remoe_target_sparsity"] = self.remoe_target_sparsity_
        return config
