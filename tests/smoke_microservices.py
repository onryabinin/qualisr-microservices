"""Check the Compose stack through the published Master and Visualization endpoints."""

import csv
import hashlib
import io
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

import httpx


DATA = Path(__file__).resolve().parents[1] / "data"
API = os.getenv("QUALISR_API_URL", "http://127.0.0.1:8080")
UI = os.getenv("QUALISR_UI_URL", "http://127.0.0.1:8501")


def expect(response, status):
    assert response.status_code == status, f"{response.request.url}: {response.text}"
    return response


def enqueue(client, **settings):
    run = expect(client.post("/experiments", json=settings), 201).json()
    assert run["status"] == "created"
    queued = expect(client.post(f"/experiments/{run['id']}/run"), 202).json()
    assert queued["status"] == "queued"
    return run["id"]


def wait_for_run(client, run_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = expect(client.get(f"/experiments/{run_id}"), 200).json()
        if run["status"] == "succeeded":
            assert run["exit_code"] == 0
            return run
        assert run["status"] != "failed", run["error"]
        time.sleep(0.5)
    raise AssertionError(f"Experiment {run_id} did not finish")


def check_artifacts(client, ui, run, expected_rows):
    run_id = run["id"]
    artifacts = expect(client.get(f"/experiments/{run_id}/artifacts"), 200).json()
    index = {artifact["path"]: artifact for artifact in artifacts}
    output = run["output_path"]

    def fetch(path):
        artifact = index[path]
        content = expect(client.get(f"/artifacts/{artifact['id']}"), 200).content
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        return content

    config = expect(client.get(f"/artifacts/{index['config.json']['id']}"), 200).json()
    assert config["provenance"]["qualisr_lab_version"] == "0.2.0"
    assert config["provenance"]["dataset_id"] == run["dataset_id"]
    assert config["provenance"]["python_version"].startswith("3.12.")
    packages = config["provenance"]["packages"]
    assert {"numpy", "pandas", "scipy", "scikit-learn"} <= packages.keys()
    assert ("xgboost" in packages) == (run["models"]["xgb"] is not None)
    assert ("catboost" in packages) == (run["models"]["catboost"] is not None)
    assert b"Saved results" in fetch("runner.log")
    assert expect(client.get(f"/experiments/{run_id}/log"), 200).content == fetch("runner.log")
    reference = None
    for model, params in run["models"].items():
        if params is None:
            continue
        assert config["models"][model]["enabled"]
        assert all(config["models"][model]["params"][key] == value for key, value in params.items())
        path = f"{output}/predictions/predictions_{model}.csv"
        predictions = list(csv.DictReader(io.StringIO(fetch(path).decode())))
        assert len({row["sample_id"] for row in predictions}) == len(predictions) == expected_rows
        points = expect(ui.get(f"/artifacts/{index[path]['id']}/predictions"), 200).json()
        assert [point["sample_id"] for point in points] == [row["sample_id"] for row in predictions]
        assignments = {point["sample_id"]: point["fold"] for point in points}
        assert reference is None or assignments == reference
        reference = assignments
        assert f"importance_{model}.csv" in {artifact["name"] for artifact in artifacts}
    assert fetch(f"{output}/correlations/correlations.png").startswith(b"\x89PNG")
    exported = expect(ui.get(f"/artifacts/{index['config.json']['id']}"), 200)
    assert exported.json() == config and "config.json" in exported.headers["content-disposition"]
    return index["config.json"]["id"]


def check_correlations(rows, n_samples, models=("randomforest",)):
    assert {(row["model"], row["source"]) for row in rows} == (
        {(model, "regressor") for model in models} | {("MUSIQ (NR)", "metric"), ("LPIPS-VGG+GT (FR)", "metric")}
    )
    for row in rows:
        assert row["n_samples"] == n_samples and row["same_samples_as_regressor"]
        assert math.isfinite(row["plcc"]) and math.isfinite(row["srcc"])


def upload(client):
    with ExitStack() as stack:
        files = [("labels", ("labels.csv", stack.enter_context((DATA / "labels.csv").open("rb")), "text/csv"))]
        for name in ("nr.csv", "fr.csv"):
            stream = stack.enter_context((DATA / "features" / name).open("rb"))
            files.append(("features", (name, stream, "text/csv")))
        response = expect(client.post("/datasets/import", data={"name": "Smoke upload"}, files=files), 201)
    return response.json()


def main():
    assert expect(httpx.get(UI + "/health"), 200).json()["service"] == "visualization"
    assert "Запустить" in expect(httpx.get(UI + "/"), 200).text
    assert any(row["id"] == "qualisr-set120" for row in expect(httpx.get(UI + "/api/datasets"), 200).json())

    with (
        httpx.Client(base_url=API, timeout=30) as client,
        httpx.Client(base_url=UI + "/api", timeout=30) as ui,
    ):
        assert expect(client.get("/health"), 200).json()["service"] == "master"
        datasets = expect(client.get("/datasets"), 200).json()
        builtin = next(dataset for dataset in datasets if dataset["id"] == "qualisr-set120")
        assert (builtin["n_samples"], builtin["n_groups"]) == (120, 40)
        batch = expect(client.get("/datasets/qualisr-set120/batches?offset=2&limit=3"), 200).json()
        assert (batch["offset"], batch["total"], len(batch["rows"])) == (2, 120, 3)
        assert batch["rows"][0]["method"] == "supir"
        batch = expect(client.get("/datasets/qualisr-set120/batches?feature=nr&limit=2"), 200).json()
        assert batch["rows"][0]["sample_id"].startswith("QualiSR-Set120/")

        expect(client.post("/experiments", json={"dataset_id": builtin["id"], "test_size": 0.3}), 422)
        for models in ({}, {"xgb": {"max_depth": "4"}}, {"catboost": {"depth": 17}}):
            expect(ui.post("/experiments", json={"dataset_id": builtin["id"], "models": models}), 422)
        defaults = expect(ui.get("/model-defaults"), 200).json()
        assert defaults["randomforest"]["n_estimators"] == 16
        created = expect(client.post("/experiments", json={"dataset_id": builtin["id"]}), 201).json()
        split_id = created["id"]
        assert expect(ui.get(f"/experiments/{split_id}"), 200).json()["status"] == "created"
        assert expect(ui.post(f"/experiments/{split_id}/run"), 202).json()["status"] == "queued"
        expect(client.post(f"/experiments/{split_id}/run"), 409)

        invalid = expect(client.post("/datasets/import", data={"name": "Invalid smoke upload"}, files=[
            ("labels", ("labels.csv", b"method,test_case\npasd,x\n", "text/csv")),
            ("features", ("nr.csv", b"sample_id,musiq\nx,1\n", "text/csv")),
        ]), 422).json()
        bad_id = invalid["detail"]["import_id"]
        assert expect(client.get(f"/imports/{bad_id}"), 200).json()["status"] == "failed"

        entry = upload(ui)
        dataset_id = entry["dataset"]["id"]
        assert expect(client.get(f"/imports/{entry['import_id']}"), 200).json()["status"] == "succeeded"
        response = expect(client.patch(f"/datasets/{dataset_id}", json={"name": "Renamed smoke upload"}), 200)
        assert response.json()["name"] == "Renamed smoke upload"
        models = {
            "randomforest": {"n_estimators": 8, "min_samples_leaf": 2},
            "xgb": {"n_estimators": 8, "max_depth": 3},
            "catboost": {"iterations": 8, "depth": 3},
        }
        cv_id = expect(ui.post("/experiments", json={
            "dataset_id": dataset_id, "mode": "cv", "n_splits": 4, "models": models,
        }), 202).json()["id"]
        expect(client.delete(f"/datasets/{dataset_id}"), 409)

        split = wait_for_run(client, split_id)
        results = expect(client.get(f"/experiments/{split_id}/results"), 200).json()
        assert results["mode"] == "split" and results["n_predictions"] == 24
        check_correlations(results["correlations"], 24)
        check_artifacts(client, ui, split, 24)
        regressor = next(row for row in results["correlations"] if row["source"] == "regressor")
        print(f"Split {split_id}: PLCC={regressor['plcc']:.4f}, SRCC={regressor['srcc']:.4f}", flush=True)

        cv = wait_for_run(client, cv_id)
        assert cv["started_at"] >= split["finished_at"], "ML ran two experiments concurrently"
        results = expect(client.get(f"/experiments/{cv_id}/results"), 200).json()
        assert results["aggregation"] == "mean_per_fold" and results["n_predictions"] == 120
        assert all(row["n_folds"] == 4 for row in results["correlations"])
        for fold in range(1, 5):
            check_correlations([row for row in results["fold_correlations"] if row["fold"] == fold], 30, models)
        assert cv_id in {run["id"] for run in expect(ui.get("/experiments"), 200).json()}
        assert "Saved results" in expect(ui.get(f"/experiments/{cv_id}/log"), 200).text
        artifact_id = check_artifacts(client, ui, cv, 120)
        expect(client.delete(f"/experiments/{cv_id}"), 204)
        expect(client.get(f"/artifacts/{artifact_id}"), 404)
        expect(client.delete(f"/datasets/{dataset_id}"), 204)
        expect(client.get(f"/datasets/{dataset_id}"), 404)
        print("CV, uploads, batches, import log, artifact downloads and CRUD OK")


if __name__ == "__main__":
    main()
