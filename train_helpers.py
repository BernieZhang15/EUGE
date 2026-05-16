import json
import os
import random
import statistics
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixlora.config import MixLoraConfig


_POSTFIX_AUX_SPECS = [
    ("load_balance_loss", "load_bal", "{:.2e}", lambda value: value > 0.0),
    ("dynmole_entropy_loss", "dent", "{:.2e}", lambda value: value > 0.0),
    ("remoe_reg_loss", "rreg", "{:.2e}", lambda value: value > 0.0),
    ("discriminative_loss", "disc", "{:.2e}", lambda value: value != 0.0),
    ("evidential_sparsity_loss", "spar", "{:.2e}", lambda value: value > 0.0),
    ("evidence_calibration_loss", "cal", "{:.2e}", lambda value: value > 0.0),
    ("ortho_loss", "ortho", "{:.2e}", lambda value: value > 0.0),
]

_POSTFIX_EUGE_SPECS = [
    ("uncertainty", "u", "{:.4f}"),
    ("exploration_coeff", "rho", "{:.4f}"),
    ("exploration_frac", "exp", "{:.3f}"),
    ("top_evidence", "etop", "{:.3f}"),
    ("tail_evidence", "etail", "{:.3f}"),
    ("tail_ratio", "tailr", "{:.3f}"),
]

_EPOCH_AUX_SPECS = [
    ("load_balance_loss", "load_bal", "{:.3e}", lambda value: value > 0.0),
    ("dynmole_entropy_loss", "dent", "{:.3e}", lambda value: value > 0.0),
    ("remoe_reg_loss", "rreg", "{:.3e}", lambda value: value > 0.0),
    ("discriminative_loss", "disc", "{:.3e}", lambda value: value != 0.0),
    ("evidential_sparsity_loss", "spar", "{:.3e}", lambda value: value > 0.0),
    ("evidence_calibration_loss", "cal", "{:.3e}", lambda value: value > 0.0),
    ("ortho_loss", "ortho", "{:.3e}", lambda value: value > 0.0),
]

_EPOCH_EUGE_SPECS = [
    ("uncertainty", "u", "{:.6f}"),
    ("exploration_coeff", "rho", "{:.6f}"),
    ("exploration_frac", "exp", "{:.6f}"),
    ("top_evidence", "etop", "{:.6f}"),
    ("tail_evidence", "etail", "{:.6f}"),
    ("tail_ratio", "tailr", "{:.6f}"),
]


def _append_metric_if_present(
    target: Dict[str, str],
    source: Dict,
    source_key: str,
    target_key: str,
    fmt: str,
    predicate,
) -> None:
    value = source.get(source_key)
    if value is None or not predicate(value):
        return
    target[target_key] = fmt.format(value)


def accumulate_scalar_stats(accumulator: Dict[str, float], stats: Dict) -> None:
    for key, value in stats.items():
        if value is None or not isinstance(value, (int, float)):
            continue
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def average_scalar_stats(accumulator: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / count for key, value in accumulator.items()}


def build_train_postfix(
    loss_value: float,
    aux_stats: Dict,
    euge_stats: Dict,
    expert_sparsity_stats: Dict,
) -> Dict[str, str]:
    postfix = {
        "loss": f"{loss_value:.4f}",
    }
    for source_key, target_key, fmt, predicate in _POSTFIX_AUX_SPECS:
        _append_metric_if_present(postfix, aux_stats, source_key, target_key, fmt, predicate)
    for source_key, target_key, fmt in _POSTFIX_EUGE_SPECS:
        _append_metric_if_present(
            postfix,
            euge_stats,
            source_key,
            target_key,
            fmt,
            lambda _: True,
        )
    _append_metric_if_present(
        postfix,
        expert_sparsity_stats,
        "active_experts_per_token",
        "aexp",
        "{:.2f}",
        lambda _: True,
    )
    _append_metric_if_present(
        postfix,
        expert_sparsity_stats,
        "routing_sparsity",
        "rspar",
        "{:.3f}",
        lambda _: True,
    )
    _append_metric_if_present(
        postfix,
        aux_stats,
        "dynmole_entropy_coef",
        "dbeta",
        "{:.2e}",
        lambda _: True,
    )
    _append_metric_if_present(
        postfix,
        aux_stats,
        "remoe_reg_coef",
        "rl1",
        "{:.2e}",
        lambda _: True,
    )
    return postfix


def build_epoch_summary_suffix(
    aux_stats: Dict,
    euge_stats: Dict,
    expert_sparsity_stats: Dict,
) -> str:
    parts = []
    for source_key, label, fmt, predicate in _EPOCH_AUX_SPECS:
        value = aux_stats.get(source_key)
        if value is not None and predicate(value):
            parts.append(f"{label} {fmt.format(value)}")
    for source_key, label, fmt in _EPOCH_EUGE_SPECS:
        value = euge_stats.get(source_key)
        if value is not None:
            parts.append(f"{label} {fmt.format(value)}")
    value = expert_sparsity_stats.get("active_experts_per_token")
    if value is not None:
        parts.append(f"aexp {value:.6f}")
    value = expert_sparsity_stats.get("routing_sparsity")
    if value is not None:
        parts.append(f"rspar {value:.6f}")
    value = aux_stats.get("remoe_reg_coef")
    if value is not None:
        parts.append(f"rl1 {value:.3e}")
    value = aux_stats.get("dynmole_entropy_coef")
    if value is not None:
        parts.append(f"dbeta {value:.3e}")
    return "".join(f" | {part}" for part in parts)


def build_lora_config(
    cfg: dict,
    dtype: torch.dtype,
    target_modules: Dict,
    routing_strategy: str,
) -> MixLoraConfig:
    default_router_bias_init = -0.02 if routing_strategy == "remoe" else 0.0
    if routing_strategy == "dynmole":
        load_balance_loss_coef = 0.0
        discriminative_loss_coef = 0.0
        evidential_sparsity_loss_coef = 0.0
        evidence_calibration_loss_coef = 0.0
        expert_ortho_loss_coef = 0.0
        dynmole_entropy_loss_coef = 1e-2
    else:
        load_balance_loss_coef = float(cfg.get("load_balance_loss_coef", 0.0))
        discriminative_loss_coef = float(cfg.get("discriminative_loss_coef", 0.0))
        evidential_sparsity_loss_coef = float(
            cfg.get("evidential_sparsity_loss_coef", 0.0)
        )
        evidence_calibration_loss_coef = float(
            cfg.get("evidence_calibration_loss_coef", 0.0)
        )
        expert_ortho_loss_coef = float(cfg.get("expert_ortho_loss_coef", 0.0))
        dynmole_entropy_loss_coef = float(
            cfg.get("dynmole_entropy_loss_coef", 1e-2)
        )

    return MixLoraConfig(
        base_model_=cfg["base_model"],
        task_type_="CAUSAL_LM",
        peft_type_="MIXLORA",
        adapter_name_="default",
        dtype_=dtype,
        lora_r_=int(cfg.get("lora_r", 8)),
        lora_alpha_=int(cfg.get("lora_alpha", 16)),
        lora_dropout_=float(cfg.get("lora_dropout", 0.05)),
        lora_init_=cfg.get("lora_init", "original"),
        target_modules_=target_modules,
        num_experts_=int(cfg.get("num_experts", 8)),
        top_k_=int(cfg.get("top_k", 2)),
        routing_strategy_=routing_strategy,
        load_balance_loss_coef_=load_balance_loss_coef,
        discriminative_loss_coef_=discriminative_loss_coef,
        evidential_sparsity_loss_coef_=evidential_sparsity_loss_coef,
        evidence_calibration_loss_coef_=evidence_calibration_loss_coef,
        expert_ortho_loss_coef_=expert_ortho_loss_coef,
        router_init_range_=float(cfg.get("router_init_range", 0.02)),
        router_bias_init_=float(cfg.get("router_bias_init", default_router_bias_init)),
        u_threshold_=float(cfg.get("u_threshold", 0.1)),
        dynmole_top_p_=float(cfg.get("dynmole_top_p", 0.75)),
        dynmole_entropy_threshold_=float(cfg.get("dynmole_entropy_threshold", 0.9)),
        dynmole_entropy_index_=float(cfg.get("dynmole_entropy_index", 1.1)),
        dynmole_entropy_loss_coef_=dynmole_entropy_loss_coef,
        remoe_reg_init_=float(cfg.get("remoe_reg_init", 1e-8)),
        remoe_reg_update_mul_=float(cfg.get("remoe_reg_update_mul", 1.2)),
        remoe_target_sparsity_=(
            None
            if cfg.get("remoe_target_sparsity") is None
            else float(cfg.get("remoe_target_sparsity"))
        ),
    )


def resolve_loss_runtime(cfg: dict, lora_cfg: MixLoraConfig) -> Dict:
    return {
        "load_balance_coef": lora_cfg.load_balance_loss_coef_,
        "discriminative_coef": lora_cfg.discriminative_loss_coef_,
        "evidential_sparsity_coef": lora_cfg.evidential_sparsity_loss_coef_,
        "expert_ortho_coef": lora_cfg.expert_ortho_loss_coef_,
        "sparsity_eps": cfg.get("sparsity_loss_eps", 1e-8),
        "evidence_calibration_coef": lora_cfg.evidence_calibration_loss_coef_,
        "evidence_eta": cfg.get("eta", 1.0),
        "evidence_loss_min": cfg.get("loss_min", 0.0),
        "evidence_loss_max": cfg.get("loss_max", 3.0),
        "router_bias_init": lora_cfg.router_bias_init_,
        "u_threshold": cfg.get("u_threshold", 0.1),
        "dynmole_top_p": lora_cfg.dynmole_top_p_,
        "dynmole_entropy_threshold": lora_cfg.dynmole_entropy_threshold_,
        "dynmole_entropy_index": lora_cfg.dynmole_entropy_index_,
        "dynmole_entropy_loss_coef": lora_cfg.dynmole_entropy_loss_coef_,
        "remoe_reg_coef": lora_cfg.remoe_reg_init_,
        "remoe_reg_update_mul": lora_cfg.remoe_reg_update_mul_,
        "remoe_target_sparsity": lora_cfg.remoe_target_sparsity_,
    }


def resolve_eval_runtime(cfg: dict) -> Dict:
    return {
        "eval_prompt_max_length": cfg.get("max_length", 512),
        "eval_batch_size": cfg.get("eval_batch_size", 8),
    }


def resolve_dataloader_runtime(cfg: dict, device: str) -> Dict[str, object]:
    num_workers_cfg = cfg.get("num_workers")
    if num_workers_cfg is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)
    else:
        num_workers = int(num_workers_cfg)
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    pin_memory_cfg = cfg.get("pin_memory")
    if pin_memory_cfg is None:
        pin_memory = device.startswith("cuda")
    else:
        pin_memory = bool(pin_memory_cfg)

    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_step(
    optimizer,
    scheduler,
    trainable_params,
    max_grad_norm,
) -> None:
    if max_grad_norm is not None and max_grad_norm > 0:
        nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def compute_per_token_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, vocab_size = logits.shape
    if sequence_length == 0:
        return logits.new_zeros((batch_size, sequence_length))

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, sequence_length - 1)

    full_loss = logits.new_zeros((batch_size, sequence_length), dtype=torch.float32)
    full_loss[..., :-1] = token_loss
    return full_loss


def compute_per_token_label_mask(labels: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length = labels.shape
    full_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
        device=labels.device,
    )
    if sequence_length == 0:
        return full_mask

    shift_labels = labels[..., 1:].contiguous()
    full_mask[..., :-1] = shift_labels.ne(-100)
    return full_mask


def resolve_seeds(cfg: dict) -> List[int]:
    seeds = cfg.get("seeds")
    if seeds is None:
        return [42]
    if isinstance(seeds, list):
        if not seeds:
            raise ValueError("'seeds' must be a non-empty integer or integer list")
        return [int(seed) for seed in seeds]
    return [int(seeds)]


def build_seed_output_dir(base_output_dir: str, seed: int, run_index: int) -> str:
    suffix = f"seed_{seed}"
    if run_index > 0:
        suffix += f"_run{run_index + 1}"
    return os.path.join(base_output_dir, suffix)


def sanitize_path_component(value: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in value
    ).strip("_")
    return sanitized or "task"


def _format_experiment_value(value) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return sanitize_path_component(str(value))


def build_experiment_dirname(cfg: dict) -> str:
    from mixlora.config import normalize_routing_strategy

    parts = [
        sanitize_path_component(
            normalize_routing_strategy(cfg.get("routing_strategy", "top-k"))
        )
    ]

    loss_name_map = [
        ("lb", "load_balance_loss_coef"),
        ("dent", "dynmole_entropy_loss_coef"),
        ("disc", "discriminative_loss_coef"),
        ("spar", "evidential_sparsity_loss_coef"),
        ("cal", "evidence_calibration_loss_coef"),
        ("ortho", "expert_ortho_loss_coef"),
    ]
    for prefix, key in loss_name_map:
        value = cfg.get(key, 0.0)
        if value not in (None, 0, 0.0):
            parts.append(f"{prefix}{_format_experiment_value(value)}")

    return "_".join(parts)


def build_task_output_dir(base_output_dir: str, task_name: str, run_index: int) -> str:
    suffix = sanitize_path_component(task_name)
    if run_index > 0:
        suffix += f"_run{run_index + 1}"
    return os.path.join(base_output_dir, suffix)


def single_task_batch_enabled(cfg: dict) -> bool:
    return bool(cfg.get("single_task_batch"))


def _compute_accuracy_stats(values: List[float]) -> Dict[str, float]:
    mean = statistics.mean(values)
    max_deviation = max((abs(value - mean) for value in values), default=0.0)
    return {
        "mean": mean,
        "max_deviation": max_deviation,
        "min": min(values),
        "max": max(values),
        "num_runs": len(values),
    }


def aggregate_multi_seed_results(run_summaries: List[Dict]) -> Dict:
    aggregated: Dict = {}
    benchmark_order: List[str] = []
    seen = set()

    for summary in run_summaries:
        final_results = summary["final_results"]
        for name, result in final_results.items():
            if "accuracy" not in result:
                continue
            aggregated.setdefault(name, []).append(
                {
                    "seed": summary["seed"],
                    "accuracy": result["accuracy"],
                    "output_dir": summary["output_dir"],
                }
            )
            if name not in seen:
                benchmark_order.append(name)
                seen.add(name)

    metrics = {}
    for name in benchmark_order:
        entries = aggregated[name]
        values = [entry["accuracy"] for entry in entries]
        metrics[name] = {
            **_compute_accuracy_stats(values),
            "values": entries,
        }

    return {
        "num_runs": len(run_summaries),
        "seeds": [summary["seed"] for summary in run_summaries],
        "metrics": metrics,
    }


def save_multi_seed_outputs(
    base_output_dir: str,
    run_summaries: List[Dict],
) -> Dict:
    summary = aggregate_multi_seed_results(run_summaries)
    summary_path = os.path.join(base_output_dir, "multi_seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    runs_path = os.path.join(base_output_dir, "multi_seed_runs.json")
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump({"runs": run_summaries}, f, indent=2)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "runs_path": runs_path,
    }


def print_multi_seed_results(summary: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"{'Benchmark':<14} {'Mean +/- MaxDev':>24} {'Runs':>8}")
    print("-" * 72)
    for name, result in summary["metrics"].items():
        print(
            f"{name:<14} "
            f"{result['mean'] * 100:>9.2f}% +/- {result['max_deviation'] * 100:>6.2f}% "
            f"{result['num_runs']:>8}"
        )
    print("=" * 72)
