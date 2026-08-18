"""Command-line interface for CardiLearn."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .artifacts import save_run
from .config import SplitConfig, TrainingConfig
from .data import load_csv
from .loaders import load_table
from .models import available_models
from .registry import ModelRegistry
from .training import train
from .validation import validate_dataset

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardilearn", description="Reproducible machine learning for cardiac datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train", help="train a model from a CSV")
    run.add_argument("--data", required=True); run.add_argument("--target", required=True); run.add_argument("--group", default=None)
    run.add_argument("--task", choices=["classification", "regression"], default="classification"); run.add_argument("--model", default="logistic_regression")
    run.add_argument("--output", required=True); run.add_argument("--seed", type=int, default=42); run.add_argument("--test-size", type=float, default=.20); run.add_argument("--validation-size", type=float, default=.20)
    run.add_argument("--no-stratify", action="store_true")
    val = sub.add_parser("validate", help="audit a dataset before training")
    val.add_argument("--data", required=True); val.add_argument("--target", required=True); val.add_argument("--id", default="sample_id"); val.add_argument("--group", default=None)
    sub.add_parser("models", help="list available baseline models").add_argument("--task", choices=["classification", "regression"], default="classification")
    sub.add_parser("runs", help="list local registered runs").add_argument("--root", default="runs")
    return parser

def main() -> int:
    args = build_parser().parse_args()
    if args.command == "models": print(json.dumps(available_models(args.task), indent=2)); return 0
    if args.command == "runs": print(json.dumps(ModelRegistry(args.root).list_runs(), indent=2)); return 0
    if args.command == "validate":
        report = validate_dataset(load_table(args.data), target=args.target, id_column=args.id, group_column=args.group)
        print(json.dumps(report.__dict__, indent=2, default=list)); return 0 if report.ok else 2
    dataset = load_csv(args.data, target_column=args.target, group_column=args.group)
    config = TrainingConfig(task=args.task, model=args.model, target_column=args.target, group_column=args.group, random_state=args.seed,
        split=SplitConfig(test_size=args.test_size, validation_size=args.validation_size, random_state=args.seed, stratify=not args.no_stratify))
    result = train(dataset, config); output = save_run(result, config, args.output)
    print(f"saved run to {Path(output).resolve()}"); print(json.dumps(result.metrics, indent=2, sort_keys=True)); return 0
