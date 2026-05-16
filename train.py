import argparse
import copy
import json
import logging
import math
import shutil
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_scheduler

from aux_loss import (
    DYNMOLE_ENTROPY_INDEX,
    DYNMOLE_ENTROPY_LOSS_COEF,
    DYNMOLE_LOAD_BALANCE_LOSS_COEF,
    DYNMOLE_LOSS_TOP_K,
    compute_aux_loss,
    summarize_router_stats,
)
from dataset import AVAILABLE_DATASETS, build_train_dataset, collate_fn
from eval_utils import infer_eval_benchmarks, log_results, print_results, run_eval
from mixlora.builder import build_mixlora_model
from mixlora.config import normalize_routing_strategy, resolve_target_modules
from mixlora.utils import (
    configure_external_log_levels,
    configure_file_logging,
    resolve_device,
    resolve_dtype,
)
from train_helpers import (
    accumulate_scalar_stats,
    average_scalar_stats,
    build_experiment_dirname,
    build_epoch_summary_suffix,
    build_lora_config,
    build_seed_output_dir,
    build_task_output_dir,
    build_train_postfix,
    compute_per_token_label_mask,
    compute_per_token_task_loss,
    optimizer_step,
    print_multi_seed_results,
    resolve_dataloader_runtime,
    resolve_eval_runtime,
    resolve_loss_runtime,
    resolve_seeds,
    save_multi_seed_outputs,
    set_seed,
    single_task_batch_enabled,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
configure_external_log_levels()
logger = logging.getLogger(__name__)

_TRAIN_TQDM_NCOLS = 220
_TRAIN_TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]"
_TRAIN_POSTFIX_UPDATE_EVERY = 10
_HAS_LOGGED_TRAINING_CONFIG = False


def _train_single(cfg: dict) -> Dict:
    global _HAS_LOGGED_TRAINING_CONFIG

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    logger.info(f"Device: {device} | dtype: {dtype}")

    output_dir = cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    log_path = configure_file_logging(output_dir)
    logger.info(f"Logging to {log_path}")
    if not _HAS_LOGGED_TRAINING_CONFIG:
        logger.info(f"Training config: {json.dumps(cfg, ensure_ascii=True, sort_keys=True)}")
        _HAS_LOGGED_TRAINING_CONFIG = True

    routing_strategy = normalize_routing_strategy(cfg.get("routing_strategy", "top-k"))
    train_datasets: List[str] = cfg.get("datasets", ["arc_c"])
    for ds in train_datasets:
        if ds not in AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset '{ds}'. Choose from {AVAILABLE_DATASETS}")
    eval_datasets: List[str] = cfg.get("eval_datasets") or infer_eval_benchmarks(train_datasets)

    base_model_config = AutoConfig.from_pretrained(cfg["base_model"])
    target_modules = resolve_target_modules(
        base_model_config.model_type,
        override=cfg.get("target_modules"),
    )

    lora_cfg = build_lora_config(cfg, dtype, target_modules, routing_strategy)
    lora_cfg.check()

    model = build_mixlora_model(
        cfg["base_model"],
        lora_cfg,
        device,
        dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    dataloader_runtime = resolve_dataloader_runtime(cfg, device)

    train_dataset = build_train_dataset(train_datasets, tokenizer, cfg.get("max_length", 512))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloader_runtime["num_workers"],
        pin_memory=dataloader_runtime["pin_memory"],
        persistent_workers=dataloader_runtime["persistent_workers"],
        drop_last=False,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.get("learning_rate", 2e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    num_epochs = cfg.get("num_epochs", 3)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    if steps_per_epoch == 0:
        raise ValueError(
            "Not enough batches for one optimizer step. "
            "Increase dataset size or reduce gradient_accumulation_steps."
        )
    total_steps = steps_per_epoch * num_epochs
    scheduler = get_scheduler("constant", optimizer=optimizer)

    max_grad_norm = cfg.get("max_grad_norm")
    loss_runtime = resolve_loss_runtime(cfg, lora_cfg)
    eval_runtime = resolve_eval_runtime(cfg)
    load_balance_coef = loss_runtime["load_balance_coef"]
    discriminative_coef = loss_runtime["discriminative_coef"]
    evidential_sparsity_coef = loss_runtime["evidential_sparsity_coef"]
    expert_ortho_coef = loss_runtime["expert_ortho_coef"]
    sparsity_eps = loss_runtime["sparsity_eps"]
    evidence_calibration_coef = loss_runtime["evidence_calibration_coef"]
    evidence_eta = loss_runtime["evidence_eta"]
    evidence_loss_min = loss_runtime["evidence_loss_min"]
    evidence_loss_max = loss_runtime["evidence_loss_max"]
    router_bias_init = loss_runtime["router_bias_init"]
    u_threshold = loss_runtime["u_threshold"]
    dynmole_top_p = loss_runtime["dynmole_top_p"]
    dynmole_entropy_threshold = loss_runtime["dynmole_entropy_threshold"]
    dynmole_entropy_index = loss_runtime["dynmole_entropy_index"]
    dynmole_entropy_coef = loss_runtime["dynmole_entropy_loss_coef"]
    remoe_reg_coef = loss_runtime["remoe_reg_coef"]
    remoe_reg_update_mul = loss_runtime["remoe_reg_update_mul"]
    remoe_target_sparsity = loss_runtime["remoe_target_sparsity"]
    eval_prompt_max_length = eval_runtime["eval_prompt_max_length"]
    eval_batch_size = eval_runtime["eval_batch_size"]

    logger.info(
        f"Training: {num_epochs} epochs | {routing_strategy} | "
        f"{batches_per_epoch} batches/epoch | "
        f"{steps_per_epoch} optimizer steps/epoch | "
        f"{total_steps} total steps"
    )
    logger.info(
        "DataLoader | num_workers: %d | pin_memory: %s | persistent_workers: %s",
        dataloader_runtime["num_workers"],
        dataloader_runtime["pin_memory"],
        dataloader_runtime["persistent_workers"],
    )
    if routing_strategy == "remoe" and load_balance_coef > 0.0:
        logger.info(
            "ReMoE uses refined L1 regularization for balancing; "
            "ignoring load_balance_loss_coef=%.6f",
            load_balance_coef,
        )
        load_balance_coef = 0.0
    if routing_strategy == "remoe" and expert_ortho_coef > 0.0:
        logger.info(
            "ReMoE does not use expert orthogonality regularization; "
            "ignoring expert_ortho_loss_coef=%.6f",
            expert_ortho_coef,
        )
        expert_ortho_coef = 0.0
    logger.info(f"Base model type: {base_model_config.model_type} | Targets: {target_modules}")
    logger.info("Scheduler: constant")
    logger.info(
        "Router setup | init_range: %.6f | bias_init: %.6f",
        lora_cfg.router_init_range_,
        router_bias_init,
    )
    if routing_strategy == "EUGE":
        logger.info("EUGE | u_threshold: %.4f", u_threshold)
    if routing_strategy == "dynmole":
        if discriminative_coef > 0.0:
            logger.info(
                "DynMoLE does not use discriminative routing loss; ignoring discriminative_loss_coef=%.6f",
                discriminative_coef,
            )
            discriminative_coef = 0.0
        if evidential_sparsity_coef > 0.0:
            logger.info(
                "DynMoLE does not use evidential sparsity loss; ignoring evidential_sparsity_loss_coef=%.6f",
                evidential_sparsity_coef,
            )
            evidential_sparsity_coef = 0.0
        if evidence_calibration_coef > 0.0:
            logger.info(
                "DynMoLE does not use evidence calibration loss; ignoring evidence_calibration_loss_coef=%.6f",
                evidence_calibration_coef,
            )
            evidence_calibration_coef = 0.0
        if expert_ortho_coef > 0.0:
            logger.info(
                "DynMoLE does not use expert orthogonality regularization; ignoring expert_ortho_loss_coef=%.6f",
                expert_ortho_coef,
            )
            expert_ortho_coef = 0.0
        logger.info(
            "DynMoLE auxiliary loss uses fixed defaults; ignoring all JSON loss coefficients."
        )
        logger.info(
            "DynMoLE router | top_p: %.4f | entropy_threshold: %.4f | entropy_index: %.4f | entropy_coef: %.6f",
            dynmole_top_p,
            dynmole_entropy_threshold,
            dynmole_entropy_index,
            dynmole_entropy_coef,
        )
        logger.info(
            "DynMoLE loss | top_k: %d | load_balance: %.6f | entropy_index: %.4f | entropy_coef: %.6f",
            DYNMOLE_LOSS_TOP_K,
            DYNMOLE_LOAD_BALANCE_LOSS_COEF,
            DYNMOLE_ENTROPY_INDEX,
            DYNMOLE_ENTROPY_LOSS_COEF,
        )
    if routing_strategy == "remoe":
        logger.info(
            "ReMoE | reg_init: %.6e | reg_update_mul: %.4f | target_sparsity: %.4f",
            remoe_reg_coef,
            remoe_reg_update_mul,
            remoe_target_sparsity,
        )
    logger.info(
        "Loss coefs | load_balance: %.6f | dynmole_entropy: %.6f | remoe_reg: %.6f | discriminative: %.6f | evidential_sparsity: %.6f | evidence_calibration: %.6f | ortho: %.6f",
        (
            DYNMOLE_LOAD_BALANCE_LOSS_COEF
            if routing_strategy == "dynmole"
            else load_balance_coef
        ),
        (
            DYNMOLE_ENTROPY_LOSS_COEF
            if routing_strategy == "dynmole"
            else dynmole_entropy_coef
        ),
        remoe_reg_coef,
        discriminative_coef,
        evidential_sparsity_coef,
        evidence_calibration_coef,
        expert_ortho_coef,
    )
    if evidential_sparsity_coef > 0.0 or evidence_calibration_coef > 0.0:
        logger.info(f"Evidential sparsity eps: {sparsity_eps:.2e}")
        logger.info(
            "Evidence calibration | coef: %.6f | eta: %.6f | clip: [%.4f, %.4f]",
            evidence_calibration_coef,
            evidence_eta,
            evidence_loss_min,
            evidence_loss_max,
        )
    logger.info(
        f"Train: {train_datasets} | Eval: {eval_datasets} | "
        f"Final eval after training"
    )

    all_eval_results: Dict = {}
    global_step = 0
    running_loss = 0.0
    running_aux_loss = 0.0
    running_batches = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        aux_stats: Dict = {}
        epoch_aux_stats_sum: Dict[str, float] = {}
        epoch_euge_stats_sum: Dict[str, float] = {}
        epoch_expert_sparsity_stats_sum: Dict[str, float] = {}
        step_router_stats_raw: Dict[str, float] = {}
        accum_count = 0
        accum_target = grad_accum
        optimizer_step_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=False,
            ncols=min(_TRAIN_TQDM_NCOLS, shutil.get_terminal_size((120, 20)).columns),
            bar_format=_TRAIN_TQDM_BAR_FORMAT,
        )

        for step, batch in enumerate(pbar):
            if accum_count == 0:
                accum_target = min(grad_accum, batches_per_epoch - step)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            task_loss_per_token = None
            task_valid_mask = None
            if evidence_calibration_coef > 0.0:
                task_loss_per_token = compute_per_token_task_loss(outputs.logits, labels)
                task_valid_mask = compute_per_token_label_mask(labels)

            aux_loss, aux_stats, batch_router_stats_raw = compute_aux_loss(
                model=model,
                routing_strategy=routing_strategy,
                num_experts=lora_cfg.num_experts_,
                top_k=lora_cfg.top_k_,
                load_balance_coef=load_balance_coef,
                discriminative_coef=discriminative_coef,
                evidential_sparsity_coef=evidential_sparsity_coef,
                expert_ortho_coef=expert_ortho_coef,
                remoe_reg_coef=remoe_reg_coef,
                device=device,
                attention_mask=attention_mask,
                sparsity_eps=sparsity_eps,
                u_threshold=u_threshold,
                evidence_calibration_coef=evidence_calibration_coef,
                evidence_eta=evidence_eta,
                evidence_loss_min=evidence_loss_min,
                evidence_loss_max=evidence_loss_max,
                task_valid_mask=task_valid_mask,
                task_loss_per_token=task_loss_per_token,
            )
            accumulate_scalar_stats(step_router_stats_raw, batch_router_stats_raw)
            if routing_strategy == "remoe":
                aux_stats["remoe_reg_coef"] = float(remoe_reg_coef)
            elif routing_strategy == "dynmole":
                aux_stats["dynmole_entropy_coef"] = float(DYNMOLE_ENTROPY_LOSS_COEF)
            accumulate_scalar_stats(epoch_aux_stats_sum, aux_stats)

            (loss + aux_loss).div(accum_target).backward()
            running_loss += loss.item()
            running_aux_loss += aux_loss.item()
            running_batches += 1
            accum_count += 1

            if accum_count == accum_target:
                optimizer_step(
                    optimizer,
                    scheduler,
                    trainable_params,
                    max_grad_norm,
                )
                global_step += 1
                accum_count = 0

                euge_stats, expert_sparsity_stats = summarize_router_stats(
                    step_router_stats_raw
                )
                if routing_strategy == "remoe":
                    current_sparsity = None
                    step_total_assignments = step_router_stats_raw.get(
                        "total_assignments",
                        0.0,
                    )
                    if step_total_assignments > 0.0:
                        current_sparsity = 1.0 - (
                            step_router_stats_raw.get("active_assignments", 0.0)
                            / step_total_assignments
                        )
                    if current_sparsity is not None:
                        if current_sparsity < remoe_target_sparsity:
                            remoe_reg_coef *= remoe_reg_update_mul
                        elif current_sparsity > remoe_target_sparsity:
                            remoe_reg_coef /= remoe_reg_update_mul
                        remoe_reg_coef = max(remoe_reg_coef, 0.0)
                        expert_sparsity_stats["routing_sparsity"] = float(current_sparsity)
                    aux_stats["remoe_reg_coef"] = float(remoe_reg_coef)
                step_router_stats_raw = {}
                accumulate_scalar_stats(epoch_euge_stats_sum, euge_stats)
                accumulate_scalar_stats(
                    epoch_expert_sparsity_stats_sum,
                    expert_sparsity_stats,
                )
                optimizer_step_count += 1

                if (
                    optimizer_step_count == 1
                    or optimizer_step_count % _TRAIN_POSTFIX_UPDATE_EVERY == 0
                    or (step + 1) == batches_per_epoch
                ):
                    pbar.set_postfix(
                        build_train_postfix(
                            loss_value=loss.item(),
                            aux_stats=aux_stats,
                            euge_stats=euge_stats,
                            expert_sparsity_stats=expert_sparsity_stats,
                        )
                    )

        pbar.close()
        avg_loss = running_loss / max(running_batches, 1)
        avg_aux = running_aux_loss / max(running_batches, 1)
        avg_aux_stats = average_scalar_stats(epoch_aux_stats_sum, running_batches)
        avg_euge_stats = average_scalar_stats(epoch_euge_stats_sum, optimizer_step_count)
        avg_expert_sparsity_stats = average_scalar_stats(
            epoch_expert_sparsity_stats_sum,
            optimizer_step_count,
        )
        extra = build_epoch_summary_suffix(
            aux_stats=avg_aux_stats,
            euge_stats=avg_euge_stats if routing_strategy == "EUGE" else {},
            expert_sparsity_stats=avg_expert_sparsity_stats,
        )
        logger.info(
            f"Epoch {epoch + 1} done | Step {global_step} | "
            f"Loss {avg_loss:.4f} | Aux {avg_aux:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e}{extra}"
        )
        running_loss = 0.0
        running_aux_loss = 0.0
        running_batches = 0

    final_adapter_dir = None

    logger.info("Running final evaluation (full splits)...")
    final_results = run_eval(
        model,
        tokenizer,
        device,
        datasets=eval_datasets,
        max_samples=None,
        output_path=None,
        prompt_max_length=eval_prompt_max_length,
        batch_size=eval_batch_size,
    )
    all_eval_results["final"] = final_results
    log_results(final_results, header="Final Eval")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)

    print_results(final_results)
    logger.info(f"Done. Outputs saved to {output_dir}")
    logger.info("Final adapter saving is disabled.")
    return {
        "seed": cfg.get("seed", 42),
        "output_dir": output_dir,
        "final_adapter_dir": final_adapter_dir,
        "final_results": final_results,
    }


def _run_multi_seed_training(cfg: dict) -> Dict:
    seeds = resolve_seeds(cfg)
    base_output_dir = cfg.get("output_dir", "./output")
    experiment_dirname = build_experiment_dirname(cfg)
    experiment_output_dir = os.path.join(base_output_dir, experiment_dirname)
    os.makedirs(base_output_dir, exist_ok=True)

    if len(seeds) == 1:
        single_cfg = copy.deepcopy(cfg)
        single_cfg["seed"] = seeds[0]
        single_cfg["output_dir"] = build_seed_output_dir(
            experiment_output_dir,
            seeds[0],
            0,
        )
        single_run = _train_single(single_cfg)
        return {
            "mode": "single_seed",
            "runs": [single_run],
            "summary": None,
        }

    logger.info("Running multi-seed experiment | seeds=%s | output_dir=%s", seeds, base_output_dir)

    run_summaries: List[Dict] = []
    seed_counts: Dict[int, int] = {}

    for run_idx, seed in enumerate(seeds, start=1):
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
        seed_run_index = seed_counts[seed] - 1
        run_cfg = copy.deepcopy(cfg)
        run_cfg["seed"] = seed
        run_cfg.pop("seeds", None)
        run_cfg["output_dir"] = build_seed_output_dir(
            experiment_output_dir,
            seed,
            seed_run_index,
        )
        logger.info(
            "Starting run %d/%d | seed=%s | output_dir=%s",
            run_idx,
            len(seeds),
            seed,
            run_cfg["output_dir"],
        )
        run_summaries.append(_train_single(run_cfg))

    saved_outputs = save_multi_seed_outputs(
        experiment_output_dir,
        run_summaries,
    )
    summary = saved_outputs["summary"]
    summary_path = saved_outputs["summary_path"]
    runs_path = saved_outputs["runs_path"]

    print_multi_seed_results(summary)
    logger.info("Multi-seed summary saved to %s", summary_path)
    logger.info("Per-run summary saved to %s", runs_path)
    return {
        "mode": "multi_seed",
        "runs": run_summaries,
        "summary": summary,
        "summary_path": summary_path,
        "runs_path": runs_path,
    }


def _run_single_task_batch(cfg: dict) -> Dict:
    datasets: List[str] = cfg.get("datasets", ["arc_c"])
    base_output_dir = cfg.get("output_dir", "./output")
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(
        "Running batch single-task experiments | tasks=%s | output_dir=%s",
        datasets,
        base_output_dir,
    )

    task_counts: Dict[str, int] = {}
    task_runs: List[Dict] = []

    for task_idx, task_name in enumerate(datasets, start=1):
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
        task_run_index = task_counts[task_name] - 1

        task_cfg = copy.deepcopy(cfg)
        task_cfg["datasets"] = [task_name]
        task_cfg["single_task_batch"] = False
        task_cfg["output_dir"] = build_task_output_dir(
            base_output_dir,
            task_name,
            task_run_index,
        )

        logger.info(
            "Starting single-task run %d/%d | dataset=%s | output_dir=%s",
            task_idx,
            len(datasets),
            task_name,
            task_cfg["output_dir"],
        )
        run_result = _run_multi_seed_training(task_cfg)
        task_runs.append(
            {
                "dataset": task_name,
                "output_dir": task_cfg["output_dir"],
                "result": run_result,
            }
        )

    return {
        "mode": "single_task_batch",
        "num_tasks": len(datasets),
        "datasets": datasets,
        "tasks": task_runs,
    }


def train(cfg: dict) -> Dict:
    if single_task_batch_enabled(cfg):
        return _run_single_task_batch(cfg)
    return _run_multi_seed_training(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--single_task_batch",
        action="store_true",
        help="Run each dataset in config.datasets as an independent single-task experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.single_task_batch:
        cfg["single_task_batch"] = True
    train(cfg)
