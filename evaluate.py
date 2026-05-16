"""
evaluate.py - standalone evaluation for saved MixLoRA adapters.

Usage:
    # Infer benchmarks from training dataset name:
    python evaluate.py --adapter ./outputs/llama_7b_arc_c/final --datasets arc_c

    # Explicit benchmark list:
    python evaluate.py --adapter ./outputs/llama_7b_arc_c/final --benchmarks arc_c boolq

    # Cap samples for quick testing:
    python evaluate.py --adapter ./outputs/llama_7b_arc_c/final --datasets arc_c --max_samples 200
"""

import argparse
import logging
import os

from dataset import AVAILABLE_DATASETS
from eval_utils import EVALUATORS, infer_eval_benchmarks, load_saved_adapter, print_results, run_eval
from mixlora.utils import configure_external_log_levels

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MixLoRA adapter")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to saved adapter directory",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE_DATASETS,
        help="Training dataset name(s) - benchmarks are auto-inferred",
    )
    group.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(EVALUATORS),
        help="Explicit benchmark list",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    configure_external_log_levels()

    args = _parse_args()

    benchmarks = args.benchmarks if args.benchmarks else infer_eval_benchmarks(args.datasets)
    logger.info(f"Benchmarks: {benchmarks}")

    logger.info(f"Loading adapter: {args.adapter}")
    model, tokenizer, device = load_saved_adapter(
        args.adapter,
        device=args.device,
    )

    output_path = args.output or os.path.join(args.adapter, "eval_standalone.json")
    results = run_eval(
        model,
        tokenizer,
        device,
        datasets=benchmarks,
        max_samples=args.max_samples,
        output_path=output_path,
        prompt_max_length=args.prompt_max_length,
        batch_size=args.batch_size,
    )
    print_results(results)
    logger.info(f"Results saved to {output_path}")
