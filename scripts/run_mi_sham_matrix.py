"""Run the MI/sham model matrix on prepared per-dataset feature tables.

Expected layout:
  data/prepared/GSE153480/sample_features.csv
  data/prepared/GSE216211/sample_features.csv

The CSVs must contain the columns specified by configs/benchmarks/mi-sham-datasets.yaml
and a target column named ``injury_label`` plus ``biological_group_id``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cardilearn.benchmark_matrix import run_model_matrix, save_matrix
from cardilearn.cardiobench import load_definition
from cardilearn.loaders import load_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CardiLearn MI/sham model matrix")
    parser.add_argument("--definition", default="configs/benchmarks/mi-sham-priority-matrix.yaml")
    parser.add_argument("--data-root", default="data/prepared")
    parser.add_argument("--output", default="runs/mi-sham-matrix/results.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "hist_gradient_boosting", "mlp"],
    )
    args = parser.parse_args()

    definition = load_definition(args.definition)
    all_results = []
    # The source matrix currently contains accession entries in a YAML registry.
    # This runner discovers only tables that physically exist in the local cache.
    for accession in ("GSE153480", "GSE216211"):
        path = Path(args.data_root) / accession / "sample_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"prepared table missing: {path}. Materialize and preprocess the GEO dataset first."
            )
        frame = load_table(path)
        all_results.extend(run_model_matrix(frame, definition, models=args.models))

    output = save_matrix(all_results, args.output)
    print(json.dumps({"output": str(output), "n_results": len(all_results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
