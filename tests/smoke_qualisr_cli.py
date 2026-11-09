"""Run and verify the two QualiSR-Lab CLI smoke experiments on local CSVs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEMPLATE = ROOT / "configs/templates/regressors.json"
MODEL = "randomforest"
PREDICTIONS = f"predictions/predictions_{MODEL}.csv"
METHODS = {"pasd": "PASD", "realesrgan": "RealESRGAN", "supir": "SUPIR"}


def prepare_fixture(directory):
    """Copy the verified labels and feature files into a temporary CLI fixture."""
    features = directory / "features"
    features.mkdir(parents=True)
    shutil.copyfile(DATA / "labels.csv", directory / "labels.csv")
    labels = pd.read_csv(directory / "labels.csv", dtype={"test_case": str})
    methods = labels["method"].map(METHODS)
    assert methods.notna().all(), "Unexpected QualiSR-Set120 method"
    sample_ids = "QualiSR-Set120/" + methods + "/" + labels["test_case"] + ".npy.gz"
    assert sample_ids.is_unique, "Duplicate method and test_case in labels"

    for name in ("nr", "fr"):
        source = DATA / "features" / f"{name}.csv"
        ids = pd.read_csv(source, usecols=["sample_id"])["sample_id"]
        assert ids.is_unique and set(ids) == set(sample_ids), f"{name}: IDs differ from labels"
        shutil.copyfile(source, features / source.name)

    return labels.assign(sample_id=sample_ids).set_index("sample_id").sort_index()


def run_cli(workdir, fixture, mode):
    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    config["experiment_name"] = mode
    config["paths"]["labels"] = str(fixture / "labels.csv")
    config["paths"]["features_root"] = str(fixture / "features")
    config["cross_validation"] = {"enabled": mode == "cv", "n_splits": 4}
    run_dir = workdir / mode
    run_dir.mkdir()
    config_path = run_dir / "config.json"
    config_text = json.dumps(config, indent=2) + "\n"
    config_path.write_text(config_text, encoding="utf-8")
    log_path = run_dir / "runner.log"
    cli = Path(sys.executable).with_name("qualisr-run-regressors")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(cli), "--config", str(config_path)],
            cwd=run_dir,
            env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(workdir / "matplotlib")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert config_path.read_text(encoding="utf-8") == config_text
    return run_dir / "outputs" / f"{mode}@pca{config['features']['pca_n']}", config


def check_validation(output, expected, metric_values):
    predictions = pd.read_csv(output / PREDICTIONS).set_index("sample_id", verify_integrity=True)
    assert set(predictions.index) == set(expected.index), "Unexpected validation sample IDs"
    labels = expected.loc[predictions.index]
    np.testing.assert_allclose(predictions["mos"], labels["score"])
    np.testing.assert_array_equal(predictions["test_case"], labels["test_case"])

    correlations = pd.read_csv(output / "correlations/correlations.csv").set_index("model")
    details = pd.read_csv(output / "correlations/correlations_by_group.csv").set_index("model")
    assert set(correlations.index) == {MODEL, *metric_values}, "Missing model or baseline"
    assert correlations.loc[MODEL, "source"] == "regressor"
    scores = {MODEL: predictions["prediction"]}
    for name, values in metric_values.items():
        assert correlations.loc[name, "source"] == "metric"
        scores[name] = values.loc[predictions.index]
    for name, values in scores.items():
        assert np.isfinite(values).all()
        assert details.loc[name, "n_samples"] == len(expected)
        # Min-max normalization leaves both correlations unchanged for increasing scores.
        expected_scores = [
            pearsonr(predictions["mos"], values).statistic,
            spearmanr(predictions["mos"], values).statistic,
        ]
        np.testing.assert_allclose(
            correlations.loc[name, ["plcc", "srcc"]].to_numpy(dtype=float), expected_scores
        )

    assert not pd.read_csv(output / f"importances/importance_{MODEL}.csv").empty
    for path in (
        "correlations/correlations.png",
        "predictions/mos_prediction_scatter_all_regressors.png",
        f"importances/importance_{MODEL}.png",
    ):
        assert (output / path).stat().st_size > 0


def main():
    assert sys.version_info[:2] == (3, 12)
    assert version("qualisr-lab") == "0.2.0"
    workdir = Path(tempfile.mkdtemp(prefix="qualisr-cli-smoke-"))
    print(f"Smoke outputs: {workdir}", flush=True)
    fixture = workdir / "fixture"
    labels = prepare_fixture(fixture)
    groups = labels["test_case"]
    assert (len(labels), groups.nunique()) == (120, 40)
    nr = pd.read_csv(fixture / "features/nr.csv").set_index("sample_id")
    fr = pd.read_csv(fixture / "features/fr.csv").set_index("sample_id")
    metrics = {"MUSIQ (NR)": nr["musiq"], "LPIPS-VGG+GT (FR)": -fr["lpips-vgg_gt"]}

    split_output, config = run_cli(workdir, fixture, "split")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config["split_seed"])
    train, test = next(split.split(labels, groups=groups))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    check_validation(split_output, labels.iloc[test], metrics)
    print(f"Split OK ({len(test)} test samples)", flush=True)

    cv_output, config = run_cli(workdir, fixture, "cv")
    n_splits = config["cross_validation"]["n_splits"]
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=config["split_seed"])
    assignments = pd.read_csv(cv_output / "metadata/cross_validation_folds.csv").set_index("sample_id")
    oof = pd.read_csv(cv_output / PREDICTIONS).set_index("sample_id", verify_integrity=True)
    assert assignments.index.is_unique and set(assignments.index) == set(oof.index) == set(labels.index)
    pd.testing.assert_series_equal(oof["fold"].sort_index(), assignments["fold"].sort_index())
    for fold, (train, validation) in enumerate(cv.split(labels, groups=groups), start=1):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[validation])
        expected = labels.iloc[validation]
        assert set(assignments.index[assignments["fold"] == fold]) == set(expected.index)
        fold_output = cv_output / f"fold_{fold:02d}"
        check_validation(fold_output, expected, metrics)
        fold_predictions = pd.read_csv(fold_output / PREDICTIONS).set_index("sample_id")
        pd.testing.assert_frame_equal(
            oof.loc[fold_predictions.index].drop(columns="fold"), fold_predictions
        )

    fold_results = pd.read_csv(cv_output / "correlations/cross_validation_folds.csv")
    assert len(fold_results) == n_splits * (len(metrics) + 1)
    for fold in range(1, n_splits + 1):
        reported = pd.read_csv(cv_output / f"fold_{fold:02d}/correlations/correlations.csv")
        rows = fold_results.loc[fold_results["fold"] == fold].drop(columns="fold")
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), reported)
    summary = pd.read_csv(cv_output / "correlations/correlations.csv").set_index(["model", "source"])
    aggregate = fold_results.groupby(["model", "source"]).agg(
        plcc=("plcc", "mean"),
        srcc=("srcc", "mean"),
        plcc_std=("plcc", "std"),
        srcc_std=("srcc", "std"),
        n_folds=("fold", "nunique"),
    )
    pd.testing.assert_frame_equal(summary.sort_index(), aggregate.sort_index())
    assert (cv_output / "correlations/correlations.png").stat().st_size > 0
    print(f"CV OK ({n_splits} folds, {len(oof)} out-of-fold predictions)")


if __name__ == "__main__":
    main()
