import json
import logging
import subprocess
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from threading import Event, Thread

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse

from shared.common import CONFIG, WORKSPACE, api, service_app
from shared.contracts import Experiment, ResourceId


LOGGER = logging.getLogger("uvicorn.error")
SETTINGS = CONFIG["ml"]
RUNS = WORKSPACE / "runs"
TEMPLATE = Path(SETTINGS["template"])


def run_settings(experiment, dataset):
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    available = dataset["features"]
    selected = experiment.features
    if selected is None:
        selected = [group for group in template["features"]["include"] if group in available]
    if not selected or len(selected) != len(set(selected)):
        raise HTTPException(422, "Select one or more distinct feature groups")
    unknown = set(selected) - available.keys()
    if unknown:
        raise HTTPException(422, f"Unknown feature groups: {', '.join(sorted(unknown))}")
    min_groups = experiment.n_splits if experiment.mode == "cv" else 2
    if dataset["n_groups"] < min_groups:
        raise HTTPException(422, f"{experiment.mode} requires at least {min_groups} test_case groups")

    used_columns = set()
    excluded = set(template["features"]["exclude_columns"])
    for group in selected:
        columns = set(available[group])
        if group == "fr":
            if len(experiment.fr_refs) != len(set(experiment.fr_refs)):
                raise HTTPException(422, "Select distinct FR references")
            suffixes = tuple(f"_{ref}" for ref in experiment.fr_refs)
            columns = {column for column in columns if column.lower().endswith(suffixes)}
        # The CLI excludes regular feature columns only after merging the CSVs.
        usable = columns - excluded
        if not usable:
            raise HTTPException(422, f"No usable columns in feature group {group}")
        if group == "fr":
            missing_refs = [ref for ref in experiment.fr_refs if not any(
                column.lower().endswith(f"_{ref}") for column in usable
            )]
            if missing_refs:
                raise HTTPException(422, f"No usable FR columns for: {', '.join(missing_refs)}")
        if group == "stats":
            columns = usable
        overlap = used_columns & columns
        if overlap:
            raise HTTPException(422, f"Feature groups share a column: {min(overlap)}")
        used_columns.update(columns)
    return template, selected


def experiment_config(run, directory):
    experiment = Experiment.model_validate({
        field: run[field] for field in Experiment.model_fields if field in run
    })
    dataset = api("storage", "GET", f"/datasets/{experiment.dataset_id}")
    config, selected = run_settings(experiment, dataset)
    data_dir = WORKSPACE / "datasets" / experiment.dataset_id
    models = {
        name: {"enabled": True, "params": params.model_dump()}
        for name, params in experiment.models if params is not None
    }
    packages = ["numpy", "pandas", "scipy", "scikit-learn"]
    packages.extend(package for model, package in (
        ("xgb", "xgboost"), ("catboost", "catboost"),
    ) if model in models)
    config.update(
        experiment_name=run["id"], seed=experiment.seed, split_seed=experiment.split_seed,
        paths={
            "labels": str(data_dir / "labels.csv"),
            "features_root": str(data_dir / "features"),
            "plots_root": str(directory / "outputs"),
        },
        cross_validation={"enabled": experiment.mode == "cv", "n_splits": experiment.n_splits},
        models=models,
        provenance={
            "qualisr_lab_version": version("qualisr-lab"), "dataset_id": dataset["id"],
            "dataset_files": dataset["files"],
            "test_size": experiment.test_size if experiment.mode == "split" else None,
            "python_version": python_version(),
            "packages": {package: version(package) for package in packages},
        },
    )
    if "catboost" in config["models"]:
        config["models"]["catboost"]["params"]["allow_writing_files"] = False
    features = config["features"]
    features.update(
        include=[group for group in selected if group != "stats"],
        include_stats="stats" in selected,
        fr_refs=experiment.fr_refs,
        feature_files={group: "{features_root}/" + f"{group}.csv" for group in dataset["features"]},
    )
    if "stats" in selected:
        features["stats_columns"] = [
            column for column in dataset["features"]["stats"] if column not in features["exclude_columns"]
        ]

    # Baselines may use a CSV that is absent from the model's feature list.
    comparison = config["correlation_metrics"]
    items = []
    for item in comparison["items"]:
        column = item["metric"]
        if item.get("reference"):
            column += f"_{item['reference']}"
        if column in dataset["features"].get(item["feature"], []):
            items.append(item)
    comparison.update(enabled=bool(items), items=items)
    return config, dataset


def correlation_table(path, config):
    table = pd.read_csv(path)
    expected = {(name, "regressor") for name in config["models"]}
    expected.update((item["label"], "metric") for item in config["correlation_metrics"]["items"])
    found = set(zip(table["model"], table["source"]))
    if found != expected or len(table) != len(expected):
        raise ValueError(f"Incomplete or duplicated correlation rows in {path.name}")
    if not np.isfinite(table[["plcc", "srcc"]]).all().all():
        raise ValueError(f"Undefined PLCC or SRCC in {path.name}")
    return table


def predictions(path):
    table = pd.read_csv(path)
    if table.empty or table["sample_id"].isna().any() or not table["sample_id"].is_unique:
        raise ValueError(f"Missing or duplicated sample IDs in {path.name}")
    if not np.isfinite(table[["mos", "prediction"]]).all().all():
        raise ValueError(f"Invalid MOS or predictions in {path.name}")
    return table


def shared_predictions(folder, models, *, cv=False):
    columns = ["mos", "fold"] if cv else ["mos"]
    reference = None
    for model in models:
        path = folder / "predictions" / f"predictions_{model}.csv"
        current = predictions(path).set_index("sample_id")[columns].sort_index()
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(f"Evaluation samples differ across regressors in {path.name}")
    return reference


def result_rows(table):
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def validation_results(folder, config, count):
    table = correlation_table(folder / "correlations/correlations.csv", config)
    details = pd.read_csv(folder / "correlations/correlations_by_group.csv")
    sizes = details.groupby(["model", "source"], as_index=False)["n_samples"].sum()
    table = table.merge(sizes, on=["model", "source"], how="left", validate="one_to_one")
    if not table["n_samples"].between(1, count).all():
        raise ValueError("Missing or invalid sample counts in correlations_by_group.csv")
    if not (table.loc[table["source"] == "regressor", "n_samples"] == count).all():
        raise ValueError("Regressor correlations do not cover all predictions")
    table["same_samples_as_regressor"] = table["n_samples"] == count
    return result_rows(table)


def parse_results(output, config, dataset):
    cv = config["cross_validation"]["enabled"]
    all_predictions = shared_predictions(output, config["models"], cv=cv)
    if not cv:
        return {
            "mode": "split", "n_predictions": len(all_predictions),
            "correlations": validation_results(output, config, len(all_predictions)),
        }

    n_splits = config["cross_validation"]["n_splits"]
    summary = correlation_table(output / "correlations/correlations.csv", config)
    if len(all_predictions) != dataset["n_samples"] or not (summary["n_folds"] == n_splits).all():
        raise ValueError("Incomplete cross-validation results")
    assignments = pd.read_csv(output / "metadata/cross_validation_folds.csv").set_index("sample_id")
    by_fold = pd.read_csv(output / "correlations/cross_validation_folds.csv")
    if len(by_fold) != len(summary) * n_splits or not assignments.index.is_unique:
        raise ValueError("Invalid cross-validation fold metadata")
    if not (assignments.groupby("test_case")["fold"].nunique() == 1).all():
        raise ValueError("A test_case occurs in multiple validation folds")
    if not assignments["fold"].sort_index().equals(all_predictions["fold"]):
        raise ValueError("OOF predictions differ from fold assignments")

    fold_results = []
    for fold in range(1, n_splits + 1):
        folder = output / f"fold_{fold:02d}"
        current = shared_predictions(folder, config["models"])
        if not current.equals(all_predictions.loc[all_predictions["fold"] == fold, ["mos"]]):
            raise ValueError(f"Unexpected prediction IDs in fold {fold}")
        fold_results.extend(
            {"fold": fold, **row} for row in validation_results(folder, config, len(current))
        )
    return {
        "mode": "cv", "aggregation": "mean_per_fold", "n_predictions": len(all_predictions),
        "correlations": result_rows(summary),
        "fold_correlations": fold_results,
    }


def invoke_cli(config_path, log, stop):
    with subprocess.Popen(
        ["qualisr-run-regressors", "--config", str(config_path)],
        cwd=config_path.parent,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        while process.poll() is None:
            if stop.wait(SETTINGS["poll_seconds"]):
                process.terminate()
                try:
                    return process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return process.wait()
        return process.returncode


def artifact_paths(directory):
    return [file.relative_to(directory).as_posix() for file in directory.rglob("*") if file.is_file()]


def save_run_status(run_id, update, stop):
    path = f"/runs/{run_id}"
    retrying = False
    while True:
        try:
            if retrying and api("storage", "GET", path)["status"] == update["status"]:
                return
            api("storage", "PATCH", path, json=update)
            return
        except HTTPException as exc:
            if exc.status_code not in {500, 502, 503, 504} or stop.is_set():
                raise
            LOGGER.warning("Could not save status %s of %s: %s", update["status"], run_id, exc.detail)
            retrying = True
            stop.wait(SETTINGS["poll_seconds"])


def execute(run, stop):
    run_id = run["id"]
    save_run_status(run_id, {"status": "running"}, stop)
    directory = RUNS / run_id
    update = {"exit_code": None}
    try:
        directory.mkdir()
        with (directory / "runner.log").open("x", encoding="utf-8") as log:
            try:
                config, dataset = experiment_config(run, directory)
                config_path = directory / "config.json"
                with config_path.open("x", encoding="utf-8") as stream:
                    json.dump(config, stream, indent=2)
                update["config_path"] = "config.json"
                exit_code = invoke_cli(config_path, log, stop)
                update["exit_code"] = exit_code
                if exit_code != 0:
                    raise RuntimeError(f"QualiSR-Lab exited with code {exit_code}. See runner.log")
                output = directory / "outputs" / f"{run_id}@pca{config['features']['pca_n']}"
                update["results"] = parse_results(output, config, dataset)
                update["output_path"] = output.relative_to(directory).as_posix()
            except (ValueError, OSError, KeyError, TypeError, RuntimeError, HTTPException) as exc:
                log.write(f"ML service error: {exc}\n")
                raise
        update["status"] = "succeeded"
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, HTTPException) as exc:
        LOGGER.error("Run %s failed: %s", run_id, exc)
        update.update(status="failed", error=str(exc))

    update["artifact_paths"] = artifact_paths(directory)
    if "runner.log" in update["artifact_paths"]:
        update["log_path"] = "runner.log"
    save_run_status(run_id, update, stop)


def worker(stop):
    while not stop.is_set():
        try:
            queued = api("storage", "GET", "/runs", params={"status": "queued"})
            if queued and not stop.is_set():
                execute(queued[0], stop)
                continue
        except HTTPException as exc:
            LOGGER.error("ML worker: %s", exc.detail)
        stop.wait(SETTINGS["poll_seconds"])


@asynccontextmanager
async def lifespan(_app):
    RUNS.mkdir(parents=True, exist_ok=True)
    for run in api("storage", "GET", "/runs", params={"status": "running"}):
        paths = artifact_paths(RUNS / run["id"])
        api(
            "storage", "PATCH", f"/runs/{run['id']}",
            json={
                "status": "failed", "error": "ML service restarted during execution",
                "artifact_paths": paths,
                "config_path": "config.json" if "config.json" in paths else None,
                "log_path": "runner.log" if "runner.log" in paths else None,
            },
        )
    stop = Event()
    thread = Thread(target=worker, args=(stop,), name="qualisr-worker")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


app = service_app("ml", lifespan=lifespan)


@app.post("/runs", status_code=201)
def create_run(experiment: Experiment):
    dataset = api("storage", "GET", f"/datasets/{experiment.dataset_id}")
    _, selected = run_settings(experiment, dataset)
    body = experiment.model_copy(update={"features": selected})
    return api("storage", "POST", "/runs", json=body.model_dump())


@app.post("/runs/{run_id}/execute", status_code=202)
def queue_run(run_id: ResourceId):
    return api("storage", "PATCH", f"/runs/{run_id}", json={"status": "queued"})


@app.get("/runs/{run_id}/status")
def get_run(run_id: ResourceId):
    return api("storage", "GET", f"/runs/{run_id}")


@app.get("/runs/{run_id}/results")
def get_results(run_id: ResourceId):
    run = get_run(run_id)
    if run["status"] != "succeeded":
        raise HTTPException(409, f"Results are unavailable for a {run['status']} experiment")
    return run["results"]


@app.get("/runs/{run_id}/log")
def get_log(run_id: ResourceId):
    get_run(run_id)
    path = RUNS / run_id / "runner.log"
    if not path.is_file():
        raise HTTPException(404, "The experiment has not produced a log")
    return FileResponse(path, media_type="text/plain")
