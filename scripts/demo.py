"""Run the reproducible QualiSR demonstration through the public API."""

import argparse
import os
import time

import httpx


def run_demo(client, dataset_id, mode):
    response = client.post("/experiments", json={
        "dataset_id": dataset_id, "features": ["nr", "fr"], "fr_refs": ["rlfn"],
        "mode": mode, "seed": 42, "split_seed": 42,
        "models": {"randomforest": {"n_estimators": 16}},
        **({"n_splits": 4} if mode == "cv" else {"test_size": 0.2}),
    })
    run = response.raise_for_status().json()
    path = f"/experiments/{run['id']}"
    print(f"\n{mode} · {run['id']}: {run['status']}", flush=True)
    client.post(path + "/run").raise_for_status()
    print("  queued", flush=True)

    status = "queued"
    deadline = time.monotonic() + 120
    while status not in {"succeeded", "failed"}:
        if time.monotonic() >= deadline:
            raise SystemExit(f"Истекло время ожидания. Проверьте {client.base_url.join(path.lstrip('/'))}")
        run = client.get(path).raise_for_status().json()
        if run["status"] != status:
            status = run["status"]
            print(f"  {status}", flush=True)
        if status in {"queued", "running"}:
            time.sleep(0.75)
    if status == "failed":
        raise SystemExit(f"{run['error']} · {client.base_url.join(path.lstrip('/') + '/log')}")

    results = client.get(path + "/results").raise_for_status().json()
    if mode == "cv":
        print(f"  Среднее по 4 фолдам · {results['n_predictions']} OOF-предсказаний")
    else:
        print(f"  Тестовая выборка · {results['n_predictions']} предсказаний")
    for row in results["correlations"]:
        folds = [
            fold for fold in results.get("fold_correlations", [])
            if (fold["model"], fold["source"]) == (row["model"], row["source"])
        ]
        count = sum(fold["n_samples"] for fold in folds) if folds else row["n_samples"]
        same = all(fold["same_samples_as_regressor"] for fold in folds) if folds else row["same_samples_as_regressor"]
        coverage = "" if same else " · неполная выборка"
        print(f"  {row['model']:<20} PLCC={row['plcc']:.4f} SRCC={row['srcc']:.4f} n={count}{coverage}")

    artifacts = client.get(path + "/artifacts").raise_for_status().json()
    config = next(item for item in artifacts if item["path"] == "config.json")
    print(f"  JSON: {client.base_url.join('artifacts/' + config['id'])}")
    print(f"  CSV и графики: {client.base_url.join(path.lstrip('/') + '/artifacts')}")
    print(f"  Журнал: {client.base_url.join(path.lstrip('/') + '/log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "split", "cv"], default="both")
    args = parser.parse_args()
    api = os.getenv("QUALISR_API_URL", "http://127.0.0.1:8080")
    try:
        with httpx.Client(base_url=api, timeout=30) as client:
            client.get("/health").raise_for_status()
            dataset = client.get("/datasets/qualisr-set120").raise_for_status().json()
            print(f"{dataset['name']} · {dataset['n_samples']} строк · {dataset['n_groups']} групп")
            for mode in ("split", "cv") if args.mode == "both" else (args.mode,):
                run_demo(client, dataset["id"], mode)
    except httpx.HTTPStatusError as exc:
        raise SystemExit(f"{exc.response.status_code} {exc.request.url}\n{exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise SystemExit(f"Ошибка HTTP: {exc.request.url} · {exc}") from exc


if __name__ == "__main__":
    main()
