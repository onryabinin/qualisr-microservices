from fastapi import File, Form, Query, Response, UploadFile

from shared.common import api, download, request, service_app, upload_parts
from shared.contracts import Experiment, Group, Name, RenameDataset, ResourceId


app = service_app("master")


@app.get("/datasets")
def list_datasets():
    return api("storage", "GET", "/datasets")


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: ResourceId):
    return api("storage", "GET", f"/datasets/{dataset_id}")


@app.patch("/datasets/{dataset_id}")
def rename_dataset(dataset_id: ResourceId, body: RenameDataset):
    return api("storage", "PATCH", f"/datasets/{dataset_id}", json=body.model_dump())


@app.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: ResourceId):
    request("storage", "DELETE", f"/datasets/{dataset_id}")
    return Response(status_code=204)


@app.post("/datasets/import", status_code=201)
def import_dataset(name: Name = Form(), labels: UploadFile = File(), features: list[UploadFile] = File()):
    return api(
        "collector", "POST", "/imports", data={"name": name},
        files=upload_parts("labels", [labels]) + upload_parts("features", features),
    )


@app.get("/imports")
def list_imports():
    return api("collector", "GET", "/imports")


@app.get("/imports/{import_id}")
def get_import(import_id: ResourceId):
    return api("collector", "GET", f"/imports/{import_id}")


@app.get("/datasets/{dataset_id}/batches")
def get_batch(
    dataset_id: ResourceId,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    feature: Group | None = None,
):
    params = {"offset": offset, "limit": limit}
    if feature is not None:
        params["feature"] = feature
    return api("collector", "GET", f"/datasets/{dataset_id}/batches", params=params)


@app.post("/experiments", status_code=201)
def create_experiment(experiment: Experiment):
    return api("ml", "POST", "/runs", json=experiment.model_dump())


@app.post("/experiments/{run_id}/run", status_code=202)
def start_experiment(run_id: ResourceId):
    return api("ml", "POST", f"/runs/{run_id}/execute")


@app.get("/experiments")
def list_experiments():
    return api("storage", "GET", "/runs")


@app.get("/experiments/{run_id}")
def get_experiment(run_id: ResourceId):
    return api("ml", "GET", f"/runs/{run_id}/status")


@app.delete("/experiments/{run_id}", status_code=204)
def delete_experiment(run_id: ResourceId):
    request("storage", "DELETE", f"/runs/{run_id}")
    return Response(status_code=204)


@app.get("/experiments/{run_id}/results")
def get_results(run_id: ResourceId):
    return api("ml", "GET", f"/runs/{run_id}/results")


@app.get("/experiments/{run_id}/log")
def get_log(run_id: ResourceId):
    return download("ml", f"/runs/{run_id}/log")


@app.get("/experiments/{run_id}/artifacts")
def list_artifacts(run_id: ResourceId):
    return api("storage", "GET", f"/runs/{run_id}/artifacts")


@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: ResourceId):
    return download("storage", f"/artifacts/{artifact_id}")
