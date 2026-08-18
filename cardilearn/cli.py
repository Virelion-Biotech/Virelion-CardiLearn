"""Command-line interface for the CardiLearn baseline trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import save_run
from .config import SplitConfig, TrainingConfig
from .data import load_csv
from .models import available_models
from .training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardilearn", description="Train reproducible cardiac ML models")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("train", help="train a model from a CSV")
    run.add_argument("--data", required=True, help="input CSV")
    run.add_argument("--target", required=True, help="target column")
    run.add_argument("--group", default=None, help="optional subject/study group column")
    run.add_argument("--task", choices=["classification", "regression"], default="classification")
    run.add_argument("--model", default="logistic_regression")
    run.add_argument("--output", required=True, help="run artifact directory")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--test-size", type=float, default=0.20)
    run.add_argument("--validation-size", type=float, default=0.20)
    run.add_argument("--no-stratify", action="store_true")

    models = sub.add_parser("models", help="list available baseline models")
    models.add_argument("--task", choices=["classification", "regression"], default="classification")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "models":
        print(json.dumps(available_models(args.task), indent=2))
        return 0

    dataset = load_csv(args.data, target_column=args.target, group_column=args.group)
    config = TrainingConfig(
        task=args.task,
        model=args.model,
        target_column=args.target,
        group_column=args.group,
        random_state=args.seed,
        split=SplitConfig(
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.seed,
            stratify=not args.no_stratify,
        ),
    )
    result = train(dataset, config)
    output = save_run(result, config, args.output)
    print(f"saved run to {Path(output).resolve()}")
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    return 0
