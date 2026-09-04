"""Assemble explicitly curated 10x samples into one sparse train-ready cohort.

The sample manifest is the biological contract. Every sample must declare its study,
subject, accession, species, assay and already-reviewed task labels. This script does
not infer conditions, maturation, cell types, or subjects from filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cardilearn.ingestion import read_10x_mtx, validate_metadata

REQUIRED_MANIFEST = (
    "accession",
    "study_id",
    "subject_id",
    "sample_id",
    "species",
    "assay",
    "cell_type",
    "maturation",
    "injury",
    "condition",
    "condition_status",
    "subject_confidence",
    "matrix_dir",
)


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    missing = [column for column in REQUIRED_MANIFEST if column not in frame.columns]
    if missing:
        raise ValueError(f"sample manifest missing columns: {missing}")
    if frame.empty:
        raise ValueError("sample manifest is empty")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must occur once in the sample manifest; cell-level rows are generated later")
    if not frame["condition_status"].eq("controlled").all():
        raise ValueError("all samples must have condition_status=controlled before assembly")
    if not frame["subject_confidence"].eq("verified").all():
        raise ValueError("all samples must have subject_confidence=verified before assembly")
    validate_metadata(frame.assign(observation_id=frame["sample_id"]))
    return frame


def _align_to_genes(matrix: sparse.csr_matrix, genes: tuple[str, ...], target: tuple[str, ...]) -> sparse.csr_matrix:
    """Reindex a sparse matrix to an explicit target gene universe without densifying."""
    if len(set(genes)) != len(genes):
        raise ValueError("input matrix contains duplicate gene IDs")
    positions = {gene: index for index, gene in enumerate(genes)}
    present_target = [
        (target_index, positions[gene])
        for target_index, gene in enumerate(target)
        if gene in positions
    ]
    if not present_target:
        raise ValueError("sample has no overlap with the target gene universe")
    target_indices = np.fromiter((pair[0] for pair in present_target), dtype=np.int64)
    source_indices = np.fromiter((pair[1] for pair in present_target), dtype=np.int64)
    selected = matrix[:, source_indices].tocoo()
    selected.col = target_indices[selected.col]
    return selected.tocsr()


def _deterministic_gene_universe(frame: pd.DataFrame) -> tuple[str, ...]:
    genes: set[str] = set()
    for row in frame.itertuples(index=False):
        expression = read_10x_mtx(row.matrix_dir)
        genes.update(expression.gene_ids)
    if not genes:
        raise ValueError("no gene identifiers were found")
    return tuple(sorted(genes))


def assemble(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame, tuple[str, ...]]:
    matrices: list[sparse.csr_matrix] = []
    observations: list[pd.DataFrame] = []
    target_genes = _deterministic_gene_universe(frame)

    for row in frame.itertuples(index=False):
        expression = read_10x_mtx(row.matrix_dir)
        aligned = _align_to_genes(expression.X, expression.gene_ids, target_genes)
        prefix = f"{row.study_id}::{row.sample_id}::"
        cell_ids = [prefix + cell for cell in expression.observation_ids]
        metadata = pd.DataFrame(
            {
                "observation_id": cell_ids,
                "accession": row.accession,
                "study_id": row.study_id,
                "subject_id": row.subject_id,
                "sample_id": row.sample_id,
                "species": row.species,
                "assay": row.assay,
                "cell_type": row.cell_type,
                "maturation": row.maturation,
                "injury": row.injury,
                "condition": row.condition,
                "condition_status": row.condition_status,
                "subject_confidence": row.subject_confidence,
                "source_observation_id": expression.observation_ids,
            }
        )
        matrices.append(aligned)
        observations.append(metadata)

    combined = sparse.vstack(matrices, format="csr", dtype=np.float32)
    metadata = pd.concat(observations, ignore_index=True)
    if metadata["observation_id"].duplicated().any():
        raise ValueError("assembled observation IDs are not unique")
    return combined, metadata, target_genes


def save_csr_npz(path: str | Path, matrix: sparse.csr_matrix, observation_ids: tuple[str, ...], genes: tuple[str, ...]) -> None:
    matrix = matrix.tocsr().astype(np.float32)
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        observation_ids=np.asarray(observation_ids, dtype=str),
        gene_ids=np.asarray(genes, dtype=str),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble curated 10x samples into sparse cohort files")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expression-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    matrix, metadata, genes = assemble(manifest)
    Path(args.expression_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_output).parent.mkdir(parents=True, exist_ok=True)
    save_csr_npz(args.expression_output, matrix, tuple(metadata["observation_id"]), genes)
    metadata.to_parquet(args.metadata_output, index=False)
    print(f"assembled observations={matrix.shape[0]} genes={matrix.shape[1]} samples={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
