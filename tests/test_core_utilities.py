"""Unit tests for CardiLearn utility modules that previously had no coverage."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardilearn import adapters, featurization, io, loaders, longitudinal, omics, provenance, vision
from cardilearn.benchmark import BenchmarkResult, BenchmarkTable
from cardilearn.model_card import ModelCard


def _classification_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "subject": [f"s{i // 2}" for i in range(n)],
            "x1": x1,
            "x2": x2,
            "label": (x1 + 0.3 * rng.normal(size=n) > 0).astype(int),
        }
    )


# adapters ---------------------------------------------------------------

def test_feature_table_validation_and_merge(tmp_path):
    rna = pd.DataFrame({"sample_id": ["a", "b", "c"], "g1": [1, 2, 3], "junk": [0, 0, 0]})
    path = tmp_path / "rna.csv"
    rna.to_csv(path, index=False)
    table = adapters.load_feature_table(path, modality="rna")
    assert table.feature_columns == ("g1", "junk")
    assert list(table.sample_ids) == ["a", "b", "c"]

    trimmed = adapters.make_feature_table(rna, modality="rna", source="x", drop_columns=["junk"])
    assert trimmed.feature_columns == ("g1",)
    ecg = adapters.make_feature_table(
        pd.DataFrame({"sample_id": ["b", "c", "d"], "qt": [0.4, 0.5, 0.6]}), modality="ecg", source="y"
    )
    merged = adapters.merge_modalities([trimmed, ecg])
    assert list(merged.columns) == ["sample_id", "rna__g1", "ecg__qt"]
    assert list(merged["sample_id"]) == ["b", "c"]
    assert len(adapters.merge_modalities([trimmed, ecg], how="outer")) == 4


@pytest.mark.parametrize(
    "frame, error",
    [
        (pd.DataFrame({"id": [1]}), KeyError),
        (pd.DataFrame({"sample_id": [None], "f": [1]}), ValueError),
        (pd.DataFrame({"sample_id": ["a", "a"], "f": [1, 2]}), ValueError),
        (pd.DataFrame({"sample_id": ["a"]}), ValueError),
    ],
)
def test_feature_table_rejects_bad_input(frame, error):
    with pytest.raises(error):
        adapters.make_feature_table(frame, modality="m", source="s")


def test_merge_requires_tables_and_log1p_guards_negatives():
    with pytest.raises(ValueError):
        adapters.merge_modalities([])
    frame = pd.DataFrame({"sample_id": ["a", "b"], "g": [0.0, np.e - 1]})
    out = adapters.expression_log1p(frame)
    assert out["g"].tolist() == pytest.approx([0.0, 1.0])
    assert frame["g"].tolist()[1] == pytest.approx(np.e - 1)  # input untouched
    with pytest.raises(ValueError):
        adapters.expression_log1p(pd.DataFrame({"sample_id": ["a"], "g": [-1.0]}))


# advanced ---------------------------------------------------------------

def test_advanced_cross_validation_calibration_and_importance():
    from cardilearn import advanced

    frame = _classification_frame()
    X, y = frame[["x1", "x2"]], frame["label"]
    plain = advanced.cross_validated_scores(LogisticRegression(), X, y, cv=3)
    grouped = advanced.cross_validated_scores(LogisticRegression(), X, y, groups=frame["subject"], cv=3)
    for result in (plain, grouped):
        assert len(result["scores"]) == 3
        assert 0.5 < result["mean"] <= 1.0
    calibrated = advanced.calibrate_classifier(LogisticRegression(), X, y)
    assert calibrated.predict_proba(X).shape == (len(X), 2)
    table = advanced.permutation_importance_table(LogisticRegression().fit(X, y), X, y, n_repeats=3)
    assert table[0]["feature"] == "x1"
    unnamed = advanced.permutation_importance_table(LogisticRegression().fit(X.values, y), X.values, y, n_repeats=2)
    assert {row["feature"] for row in unnamed} == {"feature_0", "feature_1"}


# benchmark / model card ------------------------------------------------

def test_benchmark_table_best_and_save(tmp_path):
    table = BenchmarkTable()
    with pytest.raises(ValueError):
        table.best("auc")
    table.add(BenchmarkResult("d", "t", "lr", {"auc": 0.7}))
    table.add(BenchmarkResult("d", "t", "rf", {"auc": 0.8}, test={"auc": 0.6}))
    table.add(BenchmarkResult("d", "t", "nb", {"acc": 0.9}))
    assert table.best("auc").model == "rf"
    out = tmp_path / "bench.json"
    table.save(str(out))
    saved = json.loads(out.read_text())
    assert [r["model"] for r in saved] == ["lr", "rf", "nb"]
    assert saved[1]["test"] == {"auc": 0.6}


def test_model_card_round_trip(tmp_path):
    card = ModelCard("m", "1.0", "classification", "synthetic", "research", "clinical", "grouped cv", {"auc": 0.8}, ("small n",))
    out = tmp_path / "card.json"
    card.save(str(out))
    assert json.loads(out.read_text()) == {**card.to_dict(), "limitations": ["small n"], "ethical_notes": []}


# benchmark runner + CLI -------------------------------------------------

def _definition(tmp_path, split_policy="subject"):
    path = tmp_path / "def.json"
    path.write_text(json.dumps({
        "benchmark_id": "toy", "version": "1", "task": "classification",
        "target": "label", "split_policy": split_policy, "group_key": "subject",
    }))
    return path


def test_run_benchmark_trains_and_reports(tmp_path):
    from cardilearn.benchmark_runner import run_benchmark

    data = tmp_path / "data.csv"
    _classification_frame(120).to_csv(data, index=False)
    run = run_benchmark(data, _definition(tmp_path))
    assert run.definition.benchmark_id == "toy"
    assert set(run.result.metrics) == {"train", "validation"}
    assert run.result.feature_columns == ["x1", "x2"]
    assert len(run.result.dataset_fingerprint) == 64
    with pytest.raises(ValueError, match="split manifest"):
        run_benchmark(data, _definition(tmp_path, split_policy="study"))


def test_cli_commands(tmp_path, monkeypatch, capsys):
    from cardilearn import cli

    data = tmp_path / "data.csv"
    frame = _classification_frame(120)
    frame.insert(0, "sample_id", [f"id{i}" for i in range(len(frame))])
    frame.to_csv(data, index=False)
    definition = _definition(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["cardilearn", *argv])
        code = cli.main()
        return code, capsys.readouterr().out

    code, out = run("models")
    assert code == 0 and "logistic_regression" in out
    code, out = run("benchmark-info", "--definition", str(definition))
    assert json.loads(out)["benchmark_id"] == "toy"
    bench_out = tmp_path / "bench.json"
    code, out = run("benchmark", "--data", str(data), "--definition", str(definition), "--output", str(bench_out))
    assert code == 0 and json.loads(bench_out.read_text())["benchmark_id"] == "toy"
    code, out = run("validate", "--data", str(data), "--target", "label", "--group", "subject")
    assert code in (0, 2) and json.loads(out)
    code, out = run("runs", "--root", str(tmp_path / "runs"))
    assert code == 0 and json.loads(out) == []
    code, out = run("train", "--data", str(data), "--target", "label", "--group", "subject", "--output", str(tmp_path / "run"))
    assert code == 0 and "saved run to" in out


# featurization / omics / vision ----------------------------------------

def test_featurization_summaries():
    X = np.array([[1.0, 2.0, 3.0], [np.nan, 4.0, 6.0]])
    summary = featurization.waveform_summary(X)
    assert list(summary.columns) == ["waveform_mean", "waveform_std", "waveform_min", "waveform_max", "waveform_median"]
    assert summary["waveform_mean"].tolist() == pytest.approx([2.0, 5.0])
    expr = featurization.expression_variability(np.array([[0, 2], [1, 1]]))
    assert expr["expression_nonzero_fraction"].tolist() == [0.5, 1.0]
    with pytest.raises(ValueError):
        featurization.numeric_summary(np.ones(3))
    with pytest.raises(ValueError):
        featurization.expression_variability(np.ones(3))


def test_omics_reduction_and_variance_filter():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 20))
    y = (X[:, 0] > 0).astype(int)
    assert omics.supervised_reduction("classification", k=10, n_components=3).fit_transform(X, y).shape == (40, 3)
    assert omics.supervised_reduction("regression", k=5, n_components=2).fit_transform(X, X[:, 1]).shape == (40, 2)
    for kwargs in ({"task": "clustering"}, {"task": "regression", "k": 0}):
        with pytest.raises(ValueError):
            omics.supervised_reduction(**kwargs)
    matrix = np.column_stack([np.ones(5), np.arange(5.0)])
    assert omics.variance_filter(matrix).shape == (5, 1)
    with pytest.raises(ValueError):
        omics.variance_filter(np.ones(3))


def test_vision_image_statistics(tmp_path):
    gray = vision.image_statistics(np.full((4, 5), 2.0))
    assert gray == {"height": 4.0, "width": 5.0, "channels": 1.0, "global_mean": 2.0, "global_std": 0.0}
    rgb = np.zeros((2, 2, 3))
    rgb[..., 1] = 1.0
    stats = vision.image_statistics(rgb)
    assert stats["channels"] == 3.0 and stats["channel_1_mean"] == 1.0 and stats["channel_0_mean"] == 0.0
    with pytest.raises(ValueError):
        vision.image_statistics(np.ones(4))
    PIL = pytest.importorskip("PIL.Image")
    path = tmp_path / "img.png"
    PIL.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(path)
    assert vision.load_pil(path).shape == (3, 3, 3)


# io / loaders / provenance ---------------------------------------------

def test_prediction_export_and_load(tmp_path):
    out = io.export_predictions(["a", "b"], [0, 1], output=tmp_path / "p" / "preds.csv",
                                probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]), model_name="lr", dataset_fingerprint="abc")
    frame = io.load_predictions(out)
    assert frame["probability"].tolist() == pytest.approx([0.2, 0.9])
    meta = json.loads(out.with_suffix(".csv.json").read_text())
    assert meta == {"format": io.PREDICTION_FORMAT, "model_name": "lr", "split": "test", "dataset_fingerprint": "abc", "n_predictions": 2}

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"sample_id": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        io.load_predictions(bad)
    dup = tmp_path / "dup.csv"
    pd.DataFrame({"sample_id": ["a", "a"], "prediction": [0, 1], "split": "t", "model_name": "m"}).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        io.load_predictions(dup)


def test_frozen_prediction_export(tmp_path):
    out = io.export_frozen_predictions(
        sample_ids=["a", "b"], observation_ids=["o1", "o2"], true_labels=[0, 1], predicted_labels=[0, 1],
        probabilities=[0.1, 0.9], folds=[0, 1], dataset="ds", model="m", output=tmp_path / "f.csv",
    )
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["sample_id", "observation_id", "true_label", "predicted_label", "probability", "fold", "dataset", "model"]
    assert json.loads(out.with_suffix(".csv.json").read_text())["n_predictions"] == 2
    with pytest.raises(ValueError, match="same length"):
        io.export_frozen_predictions(sample_ids=["a"], observation_ids=[], true_labels=[0], predicted_labels=[0],
                                     probabilities=[0.1], folds=[0], dataset="d", model="m", output=tmp_path / "x.csv")


def test_loaders_formats_and_columns(tmp_path):
    frame = pd.DataFrame({"sample_id": ["a", "b"], "y": [0, 1], "f": [1.0, 2.0]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    frame.to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    for name in ("t.csv", "t.tsv"):
        pd.testing.assert_frame_equal(loaders.load_table(tmp_path / name), frame)
    with pytest.raises(FileNotFoundError):
        loaders.load_table(tmp_path / "missing.csv")
    (tmp_path / "t.xlsx").write_text("")
    with pytest.raises(ValueError):
        loaders.load_table(tmp_path / "t.xlsx")
    X, y = loaders.load_feature_matrix(tmp_path / "t.csv", "y", id_columns=["sample_id"])
    assert list(X.columns) == ["f"] and y.tolist() == [0, 1]
    with pytest.raises(ValueError):
        loaders.load_feature_matrix(tmp_path / "t.csv", "nope")
    messy = pd.DataFrame([[1, 2, 3]], columns=[" a b ", "c", "c"])
    assert list(loaders.canonicalize_columns(messy).columns) == ["a_b", "c"]


def test_loaders_columnar_formats(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"a": [1, 2]})
    frame.to_parquet(tmp_path / "t.parquet")
    frame.to_feather(tmp_path / "t.feather")
    for name in ("t.parquet", "t.feather"):
        pd.testing.assert_frame_equal(loaders.load_table(tmp_path / name), frame)


def test_provenance_fingerprint_is_column_order_invariant():
    a = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})
    assert provenance.dataframe_fingerprint(a) == provenance.dataframe_fingerprint(a[["y", "x"]])
    assert provenance.dataframe_fingerprint(a) != provenance.dataframe_fingerprint(a.assign(x=[1, 3]))
    meta = provenance.runtime_metadata()
    assert set(meta) == {"created_at_utc", "python", "platform", "cardilearn_version"}


# longitudinal / selection / singlecell ---------------------------------

def test_temporal_split_ungrouped_and_grouped():
    frame = pd.DataFrame({
        "t": ["2024-01-05", "2024-01-01", "2024-01-04", "2024-01-02", "2024-01-03"],
        "g": ["a", "a", "b", "c", "c"],
    })
    dev, test = longitudinal.temporal_split(frame, time_column="t", test_fraction=0.2)
    assert test.tolist() == [0] and dev.tolist() == [1, 2, 3, 4]
    dev, test = longitudinal.temporal_split(frame, time_column="t", test_fraction=0.3, group_column="g")
    assert test.tolist() == [0, 1] and dev.tolist() == [2, 3, 4]
    with pytest.raises(KeyError):
        longitudinal.temporal_split(frame, time_column="missing")
    with pytest.raises(KeyError):
        longitudinal.temporal_split(frame, time_column="t", group_column="missing")
    with pytest.raises(ValueError):
        longitudinal.temporal_split(frame, time_column="t", test_fraction=1.0)
    with pytest.raises(ValueError):
        longitudinal.temporal_split(pd.DataFrame({"t": ["not a date"]}), time_column="t")


def test_select_model_uses_cross_validation():
    from sklearn.dummy import DummyClassifier

    from cardilearn.selection import select_model

    frame = _classification_frame(80)
    X, y = frame[["x1", "x2"]], frame["label"]
    candidates = {"dummy": DummyClassifier(strategy="most_frequent"), "lr": LogisticRegression()}
    metric = "balanced_accuracy"
    best, scores = select_model(candidates, X, y, task="classification", primary_metric=metric, n_splits=3)
    assert best == "lr" and [s.name for s in scores] == ["lr", "dummy"]
    with pytest.raises(ValueError):
        select_model({}, X, y, task="classification", primary_metric=metric)
    with pytest.raises(KeyError):
        select_model(candidates, X, y, task="classification", primary_metric="no_such_metric", n_splits=3)


class _FakeAnnData:
    def __init__(self, X, obs, var_names, layers=None):
        self.X, self.obs, self.var_names, self.layers = X, obs, var_names, layers or {}


@pytest.mark.parametrize("as_sparse", [False, True])
def test_pseudobulk_counts_dense_and_sparse(as_sparse):
    from scipy import sparse

    from cardilearn import singlecell

    X = np.array([[1, 0], [2, 1], [0, 5]], dtype=float)
    obs = pd.DataFrame({"donor": ["d1", "d1", "d2"]}, index=["c1", "c2", "c3"])
    adata = _FakeAnnData(sparse.csr_matrix(X) if as_sparse else X, obs, ["g1", "g2"], layers={"raw": X})
    out = singlecell.pseudobulk_counts(adata, group_column="donor")
    assert out["donor"].tolist() == ["d1", "d2"]
    assert out[["g1", "g2"]].to_numpy().tolist() == [[3, 1], [0, 5]]
    assert singlecell.pseudobulk_counts(adata, group_column="donor", layer="raw")["g2"].tolist() == [1, 5]
    with pytest.raises(KeyError):
        singlecell.pseudobulk_counts(adata, group_column="missing")
    assert singlecell.obs_table(adata)["sample_id"].tolist() == ["c1", "c2", "c3"]
    adata.obs.index = ["c1", "c1", "c3"]
    with pytest.raises(ValueError):
        singlecell.obs_table(adata)
