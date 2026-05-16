"""
eval_utils.py - Evaluation utilities aligned with TUDB-Labs/MoE-PEFT.

Key alignment points:
  - Prompt templates follow MoE-PEFT's QA task definitions.
  - Multiple-choice evaluation uses last-token label classification instead of
    full continuation scoring.
  - Metrics are plain accuracy for these commonsense QA tasks.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import (
    COMMONSENSE_DATASET_CANDIDATES,
    _load_hf_split,
    format_commonsense_example,
)
from mixlora.model import MixLoraModelForCausalLM
from mixlora.utils import collect_expert_sparsity_stats_with_mask, infer_device

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 8

TRAIN_DATASET_TO_BENCHMARKS = {
    "arc_c": ["arc_c"],
    "arc_e": ["arc_e"],
    "boolq": ["boolq"],
    "obqa": ["obqa"],
    "piqa": ["piqa"],
    "siqa": ["siqa"],
    "hellaswag": ["hellaswag"],
    "winogrande": ["winogrande"],
}


def infer_eval_benchmarks(train_datasets: List[str]) -> List[str]:
    seen, benchmarks = set(), []
    for ds in train_datasets:
        for bm in TRAIN_DATASET_TO_BENCHMARKS.get(ds, []):
            if bm not in seen:
                benchmarks.append(bm)
                seen.add(bm)
    return benchmarks


def _infer_eval_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_saved_adapter(
    adapter_path: str,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, object, str]:
    resolved_device = device or infer_device()
    model, config = MixLoraModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=_infer_eval_dtype(resolved_device),
        device_map=resolved_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, resolved_device


def _get_label_token_ids(tokenizer, labels: Sequence[str]) -> List[int]:
    token_ids = []
    for label in labels:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not tokenize label '{label}'")
        token_ids.append(ids[-1])
    return token_ids


def _last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D, got shape {tuple(attention_mask.shape)}")

    if attention_mask.size(1) == 0:
        raise ValueError("attention_mask must have non-zero sequence length")

    valid = attention_mask.to(dtype=torch.bool)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("Found empty sequence in attention_mask")

    seq_len = attention_mask.size(1)
    from_right = valid.flip(dims=(1,)).to(dtype=torch.long).argmax(dim=1)
    return (seq_len - 1) - from_right


def _evaluate_label_task(
    model,
    tokenizer,
    device: str,
    task_name: str,
    examples: Sequence[Tuple],
    labels: Sequence[str],
    prompt_max_length: int,
    batch_size: int,
) -> Dict:
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    label_token_ids = torch.tensor(
        _get_label_token_ids(tokenizer, labels),
        dtype=torch.long,
        device=device,
    )

    correct = 0
    total = len(examples)
    sparsity_totals = {
        "active_experts_per_token": 0.0,
    }
    sparsity_counts = {
        "active_experts_per_token": 0,
    }

    for start in tqdm(range(0, total, batch_size), desc=f"  {task_name}", leave=False):
        batch = examples[start:start + batch_size]
        prompts = [item[0] for item in batch]
        gold = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            logits = model(**enc).logits
        expert_sparsity_stats = collect_expert_sparsity_stats_with_mask(
            model,
            attention_mask=enc["attention_mask"],
        )
        batch_weight = int(enc["attention_mask"].sum().item())
        for key in sparsity_totals:
            value = expert_sparsity_stats.get(key)
            if value is None:
                continue
            sparsity_totals[key] += value * batch_weight
            sparsity_counts[key] += batch_weight

        last_positions = _last_nonpad_positions(enc["attention_mask"])
        pooled = logits[torch.arange(logits.size(0), device=device), last_positions]
        choice_logits = pooled.index_select(dim=-1, index=label_token_ids)

        if len(batch[0]) >= 3:
            max_candidates = max(len(item[2]) for item in batch)
            candidate_indices = torch.zeros(
                (len(batch), max_candidates),
                dtype=torch.long,
                device=device,
            )
            candidate_mask = torch.zeros(
                (len(batch), max_candidates),
                dtype=torch.bool,
                device=device,
            )
            for row_idx, item in enumerate(batch):
                local_candidates = torch.tensor(
                    item[2],
                    dtype=torch.long,
                    device=device,
                )
                candidate_indices[row_idx, : local_candidates.numel()] = local_candidates
                candidate_mask[row_idx, : local_candidates.numel()] = True
            candidate_logits = choice_logits.gather(dim=1, index=candidate_indices)
            candidate_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))
            local_pred = candidate_logits.argmax(dim=-1, keepdim=True)
            pred = candidate_indices.gather(dim=1, index=local_pred).squeeze(1)
        else:
            pred = choice_logits.argmax(dim=-1)

        correct += int((pred == gold).sum().item())

    acc = correct / total if total else 0.0
    logger.info("  %s: %.2f%%  (%d/%d)", task_name, acc * 100, correct, total)
    result = {"accuracy": acc, "correct": correct, "total": total}
    for key, total_value in sparsity_totals.items():
        count = sparsity_counts[key]
        if count > 0:
            result[key] = total_value / count
    return result


def _maybe_cap(ds, max_samples: Optional[int]):
    if max_samples is None:
        return ds
    return ds.select(range(min(max_samples, len(ds))))


def _build_arc_examples(ds) -> Tuple[List[str], List[Tuple[str, int]]]:
    label_space = ["1", "2", "3", "4", "5", "A", "B", "C", "D", "E"]
    label_to_idx = {label: idx for idx, label in enumerate(label_space)}
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("arc_c", ex)
        if formatted is None:
            continue
        answer_key = formatted["response"]
        choices = ex["choices"]
        candidate_indices = []
        for label in choices["label"]:
            if label not in label_to_idx:
                continue
            candidate_indices.append(label_to_idx[label])
        if not candidate_indices or label_to_idx[answer_key] not in candidate_indices:
            continue
        examples.append(
            (
                formatted["prompt"],
                label_to_idx[answer_key],
                candidate_indices,
            )
        )
    return label_space, examples


def _eval_arc_split(
    dataset_key: str,
    task_name: str,
    model,
    tokenizer,
    device,
    max_samples,
    prompt_max_length,
    batch_size,
) -> Dict:
    ds = _load_hf_split(dataset_key, COMMONSENSE_DATASET_CANDIDATES[dataset_key], "test")
    ds = _maybe_cap(ds, max_samples)
    labels, examples = _build_arc_examples(ds)
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        task_name,
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_arc_c(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    return _eval_arc_split(
        "arc_c",
        "ARC-C",
        model,
        tokenizer,
        device,
        max_samples,
        prompt_max_length,
        batch_size,
    )


def eval_arc_e(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    return _eval_arc_split(
        "arc_e",
        "ARC-E",
        model,
        tokenizer,
        device,
        max_samples,
        prompt_max_length,
        batch_size,
    )


def eval_boolq(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("boolq", COMMONSENSE_DATASET_CANDIDATES["boolq"], "validation")
    ds = _maybe_cap(ds, max_samples)
    labels = ["true", "false"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("boolq", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], label_to_idx[formatted["response"]]))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "BoolQ",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_obqa(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("obqa", COMMONSENSE_DATASET_CANDIDATES["obqa"], "test")
    ds = _maybe_cap(ds, max_samples)
    labels = ["A", "B", "C", "D"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("obqa", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], label_to_idx[formatted["response"]]))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "OBQA",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_piqa(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("piqa", COMMONSENSE_DATASET_CANDIDATES["piqa"], "validation")
    ds = _maybe_cap(ds, max_samples)
    labels = ["A", "B"]
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("piqa", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], 0 if formatted["response"] == "A" else 1))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "PIQA",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_siqa(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("siqa", COMMONSENSE_DATASET_CANDIDATES["siqa"], "validation")
    ds = _maybe_cap(ds, max_samples)
    labels = ["A", "B", "C"]
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("siqa", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], ord(formatted["response"]) - ord("A")))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "SIQA",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_hellaswag(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("hellaswag", COMMONSENSE_DATASET_CANDIDATES["hellaswag"], "validation")
    ds = _maybe_cap(ds, max_samples)
    labels = ["A", "B", "C", "D"]
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("hellaswag", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], ord(formatted["response"]) - ord("A")))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "HellaSwag",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


def eval_winogrande(model, tokenizer, device, max_samples, prompt_max_length, batch_size, **_) -> Dict:
    ds = _load_hf_split("winogrande", COMMONSENSE_DATASET_CANDIDATES["winogrande"], "validation")
    ds = _maybe_cap(ds, max_samples)
    labels = ["A", "B"]
    examples = []
    for ex in ds:
        formatted = format_commonsense_example("winogrande", ex)
        if formatted is None:
            continue
        examples.append((formatted["prompt"], 0 if formatted["response"] == "A" else 1))
    return _evaluate_label_task(
        model,
        tokenizer,
        device,
        "WinoGrande",
        examples,
        labels,
        prompt_max_length,
        batch_size,
    )


EVALUATORS: Dict = {
    "arc_c": eval_arc_c,
    "arc_e": eval_arc_e,
    "boolq": eval_boolq,
    "obqa": eval_obqa,
    "piqa": eval_piqa,
    "siqa": eval_siqa,
    "hellaswag": eval_hellaswag,
    "winogrande": eval_winogrande,
}


def run_eval(
    model,
    tokenizer,
    device: str,
    datasets: List[str],
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    prompt_max_length: int = _DEFAULT_PROMPT_MAX_LENGTH,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    prediction_dir: Optional[str] = None,
) -> Dict:
    del prediction_dir

    was_training = model.training
    model.eval()

    results: Dict = {}
    logger.info("=" * 55)
    logger.info(
        "Evaluation | benchmarks: %s | max_samples: %s | batch_size: %s",
        datasets,
        max_samples,
        batch_size,
    )

    for name in datasets:
        fn = EVALUATORS.get(name)
        if fn is None:
            logger.warning("  No evaluator for '%s', skipping.", name)
            continue
        try:
            results[name] = fn(
                model,
                tokenizer,
                device,
                max_samples,
                prompt_max_length,
                batch_size,
            )
        except Exception as exc:
            logger.error("  %s eval failed: %s", name, exc)
            results[name] = {"error": str(exc)}

    accs = [r["accuracy"] for r in results.values() if "accuracy" in r]
    if len(accs) > 1:
        avg = sum(accs) / len(accs)
        results["average"] = {"accuracy": avg}
        logger.info("  Average: %.2f%%", avg * 100)

    logger.info("=" * 55)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if was_training:
        model.train()

    return results


def format_results_table(results: Dict) -> str:
    lines = [
        "=" * 48,
        f"{'Benchmark':<14} {'Accuracy':>10}",
        "-" * 48,
    ]
    for name, res in results.items():
        if name == "average":
            continue
        if "error" in res:
            lines.append(f"{name:<14} {'ERROR':>10}")
        elif "accuracy" in res:
            lines.append(f"{name:<14} {res['accuracy'] * 100:>9.2f}%")
    if "average" in results and "accuracy" in results["average"]:
        lines.append("-" * 48)
        lines.append(f"{'Average':<14} {results['average']['accuracy'] * 100:>9.2f}%")
    lines.append("=" * 48)
    return "\n".join(lines)


def log_results(results: Dict, header: Optional[str] = None) -> None:
    table = format_results_table(results)
    message = f"{header}\n{table}" if header else f"\n{table}"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        if logging.INFO < handler.level:
            continue
        record = logger.makeRecord(
            name=logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        handler.handle(record)


def print_results(results: Dict) -> None:
    print("\n" + format_results_table(results))
