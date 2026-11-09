"""Check failure handling, concurrent submissions and persistence in a disposable stack."""

import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx

from smoke_microservices import API, DATA, UI, enqueue, expect, upload, wait_for_run


BUILTIN = "qualisr-set120"


def import_files(client, name, labels, features):
    return client.post("/datasets/import", data={"name": name}, files=[
        ("labels", ("labels.csv", labels, "text/csv")),
        *(("features", (f"{group}.csv", content, "text/csv")) for group, content in features.items()),
    ])


def check_rejections(client, ui):
    labels = b"method,test_case,score\npasd,case-a,0.2\n"
    key = b"QualiSR-Set120/PASD/case-a.npy.gz"
    row = key + b",0.4\n"
    nr = b"sample_id,musiq\n" + row
    datasets = expect(client.get("/datasets"), 200).json()
    runs = expect(client.get("/experiments"), 200).json()

    for label_csv, feature_csv, message in [
        (b"", nr, "empty or duplicate CSV headers"),
        (b"method,test_case,score\n", nr, "labels.csv is empty"),
        (labels.replace(b"0.2", b"nan"), nr, "score must be finite"),
        (labels.replace(b"0.2", b"True"), nr, "score must be finite and numeric"),
        (labels.replace(b"case-a", b"case-a\0extra"), nr, "NUL byte"),
        (labels, nr.replace(b"musiq", b"musiq\0extra"), "NUL byte"),
        (labels, b'sample_id,musiq\n"unfinished', "unexpected end of data"),
        (labels, nr + row, "duplicate sample_id"),
        (labels, nr.replace(key, b"unknown"), "unknown or duplicate sample_id"),
        (labels, nr.replace(b"0.4", b"bad"), "invalid number"),
        (labels, b"sample_id,musiq\n", "missing 1 sample IDs"),
    ]:
        error = expect(import_files(client, "Invalid CSV", label_csv, {"nr": feature_csv}), 422).json()["detail"]
        assert message in error["error"], error
        entry = expect(client.get(f"/imports/{error['import_id']}"), 200).json()
        assert entry["status"] == "failed" and entry["error"] == error["error"]

    assert expect(client.get("/datasets"), 200).json() == datasets
    batches = f"/datasets/{BUILTIN}/batches"
    offset = 2**80
    assert expect(client.get(batches, params={"feature": "nr", "offset": offset}), 200).json() == {
        "offset": offset, "limit": 50, "total": 120, "rows": [],
    }
    expect(client.get(batches, params={"feature": "missing", "offset": offset}), 404)

    for patch in (
        {"features": ["missing"]}, {"features": ["nr", "nr"]}, {"features": []},
        {"models": {}}, {"models": {"unknown": {}}}, {"models": {"xgb": {"max_depth": "3"}}},
        {"models": {"catboost": {"depth": 17}}}, {"models": {"randomforest": {"n_estimators": 0}}},
        {"test_size": 0.25}, {"mode": "cv", "n_splits": 1}, {"mode": "cv", "n_splits": 41},
        {"mode": "other"}, {"split_seed": -1}, {"split_sed": 10},
        {"fr_refs": ["rlfn", "rlfn"]},
    ):
        expect(client.post("/experiments", json={"dataset_id": BUILTIN, **patch}), 422)

    for endpoint in (client, ui):
        response = expect(endpoint.post(
            "/experiments",
            content=b'{"dataset_id":"qualisr-set120","models":{"xgb":{"learning_rate":1e400}}}',
            headers={"Content-Type": "application/json"},
        ), 422)
        assert response.json()["detail"][0]["loc"] == ["body", "models", "xgb", "learning_rate"]

    expect(client.post("/experiments", json={"dataset_id": "0" * 32}), 404)
    entry = expect(import_files(client, "Single group", labels, {"nr": nr}), 201).json()
    dataset_id = entry["dataset"]["id"]
    for mode in ("split", "cv"):
        response = expect(client.post("/experiments", json={"dataset_id": dataset_id, "mode": mode}), 422)
        assert "test_case groups" in response.json()["detail"]
    expect(client.delete(f"/datasets/{dataset_id}"), 204)

    two_groups = labels + b"pasd,case-b,0.5\n"
    two_rows = nr + row.replace(b"case-a", b"case-b")
    with_excluded = two_rows.replace(b"musiq", b"musiq,qalign").replace(b",0.4\n", b",0.4,0.2\n")
    entry = expect(import_files(client, "Incompatible groups", two_groups, {
        "nr": with_excluded, "copy": two_rows, "fr": two_rows.replace(b"musiq", b"lpips-vgg_gt"),
        "shared_exclusion": with_excluded.replace(b"musiq", b"other"),
    }), 201).json()
    dataset_id = entry["dataset"]["id"]
    for settings, message in [
        ({"features": ["nr", "copy"]}, "Feature groups share a column: musiq"),
        ({"features": ["nr", "shared_exclusion"]}, "Feature groups share a column: qalign"),
        ({"features": ["fr"]}, "No usable columns in feature group fr"),
        ({"features": ["fr"], "fr_refs": ["gt", "rlfn"]}, "No usable FR columns for: rlfn"),
    ]:
        response = expect(client.post("/experiments", json={"dataset_id": dataset_id, **settings}), 422)
        assert message in response.json()["detail"]
    expect(client.delete(f"/datasets/{dataset_id}"), 204)
    assert expect(client.get("/experiments"), 200).json() == runs
    print("CSV, feature, model, protocol and missing-data rejections OK", flush=True)


def wait_for_failure(client, run_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = expect(client.get(f"/experiments/{run_id}"), 200).json()
        if run["status"] == "failed":
            return run
        assert run["status"] != "succeeded", run
        time.sleep(0.5)
    raise AssertionError(f"Failed experiment {run_id} did not finish")


def check_cli_failures(client):
    labels = (DATA / "labels.csv").read_bytes()
    with (DATA / "features" / "nr.csv").open(newline="") as source:
        ids = [row["sample_id"] for row in csv.DictReader(source)]
    failures = []
    for name, values, exit_zero, message in [
        ("No finite train values", [None] * len(ids), False, "training split has no finite values"),
        ("Constant predictions", [1] * len(ids), True, "Undefined PLCC or SRCC"),
    ]:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "feature"])
        writer.writerows(zip(ids, values, strict=True))
        entry = expect(import_files(client, name, labels, {"nr": stream.getvalue()}), 201).json()
        run_id = enqueue(client, dataset_id=entry["dataset"]["id"])
        run = wait_for_failure(client, run_id)
        assert isinstance(run["exit_code"], int) and (run["exit_code"] == 0) == exit_zero
        assert run["results"] is None and run["error"]
        expect(client.get(f"/experiments/{run_id}/results"), 409)
        assert message in expect(client.get(f"/experiments/{run_id}/log"), 200).text
        failures.append(run_id)
    print("CLI failure and exit 0 with undefined correlations both recorded as failed", flush=True)
    return failures


def check_concurrent_runs(client, dataset_id):
    ids = [
        expect(client.post("/experiments", json={
            "dataset_id": dataset_id, "seed": seed, "split_seed": 42,
            "models": {"randomforest": {"n_estimators": trees}},
        }), 201).json()["id"]
        for seed, trees in ((42, 16), (17, 8))
    ]
    barrier = Barrier(3)

    def start(run_id):
        barrier.wait(timeout=10)
        return client.post(f"/experiments/{run_id}/run")

    with ThreadPoolExecutor(max_workers=3) as pool:
        requests = [pool.submit(start, run_id) for run_id in [ids[0], ids[0], ids[1]]]
        responses = [request.result() for request in requests]
    assert sorted(response.status_code for response in responses[:2]) == [202, 409]
    expect(responses[2], 202)
    completed = sorted((wait_for_run(client, run_id) for run_id in ids), key=lambda run: run["queued_at"])
    assert completed[0]["finished_at"] <= completed[1]["started_at"]
    assert all(run["id"] in run["output_path"] for run in completed)
    assert len({run["output_path"] for run in completed}) == len(ids)
    print("Concurrent requests OK, duplicate start rejected and FIFO preserved", flush=True)
    return ids


def snapshot(client, ui, dataset_id, import_id, run_ids):
    dataset = expect(client.get(f"/datasets/{dataset_id}"), 200).json()
    batch = expect(client.get(f"/datasets/{dataset_id}/batches?feature=nr&limit=500"), 200).json()
    entry = expect(client.get(f"/imports/{import_id}"), 200).json()
    history = {run["id"]: run for run in expect(ui.get("/experiments"), 200).json()}
    saved = {"dataset": dataset, "batch": batch, "import": entry, "runs": {}}
    artifact_ids = set()
    prediction_hashes = set()
    for run_id in run_ids:
        run = expect(client.get(f"/experiments/{run_id}"), 200).json()
        assert history[run_id] == run
        files = expect(client.get(f"/experiments/{run_id}/artifacts"), 200).json()
        by_path = {item["path"]: item for item in files}
        assert len(by_path) == len(files)
        for item in files:
            assert item["id"] not in artifact_ids and item["run_id"] == run_id
            artifact_ids.add(item["id"])
            payload = expect(ui.get(f"/artifacts/{item['id']}"), 200).content
            assert len(payload) == item["size"]
            assert hashlib.sha256(payload).hexdigest() == item["sha256"], item["path"]
        config = expect(ui.get(f"/artifacts/{by_path['config.json']['id']}"), 200).json()
        assert config["experiment_name"] == run_id and config["seed"] == run["seed"]
        assert config["paths"]["plots_root"].endswith(f"/{run_id}/outputs")
        log = expect(ui.get(f"/experiments/{run_id}/log"), 200).content
        assert hashlib.sha256(log).hexdigest() == by_path["runner.log"]["sha256"]
        if run["status"] == "succeeded":
            path = f"{run['output_path']}/predictions/predictions_randomforest.csv"
            prediction_hashes.add(by_path[path]["sha256"])
            assert expect(client.get(f"/experiments/{run_id}/results"), 200).json() == run["results"]
        saved["runs"][run_id] = {"run": run, "artifacts": files}
    assert len(prediction_hashes) == 2, "Both concurrent runs must have distinct prediction files"
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Path to a new snapshot JSON, or an existing one with --verify")
    parser.add_argument("--verify", action="store_true", help="Compare state after recreating the containers")
    args = parser.parse_args()

    with (
        httpx.Client(base_url=API, timeout=30) as client,
        httpx.Client(base_url=UI + "/api", timeout=30) as ui,
    ):
        if args.verify:
            before = json.loads(args.snapshot.read_text(encoding="utf-8"))
            after = snapshot(client, ui, before["dataset"]["id"], before["import"]["id"], before["runs"])
            assert before == after, "Dataset, import, history or artifact metadata changed after recreation"
            print("Dataset, import, four runs and all artifact bytes persisted after recreation")
            return

        if args.snapshot.exists():
            parser.error("Snapshot already exists. Choose a new path.")
        check_rejections(client, ui)
        failed = check_cli_failures(client)
        entry = upload(client)
        dataset_id = entry["dataset"]["id"]
        successful = check_concurrent_runs(client, dataset_id)
        saved = snapshot(client, ui, dataset_id, entry["import_id"], successful + failed)
        with args.snapshot.open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, indent=2, ensure_ascii=False)
        print(f"Separate configs, logs and outputs verified. Snapshot: {args.snapshot}")


if __name__ == "__main__":
    main()
