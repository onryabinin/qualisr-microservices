import csv
import io
import math
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from shared.common import api, download, request, service_app, upload_parts
from shared.contracts import (
    CatBoostParams, Experiment, ExperimentModels, Name, RandomForestParams, ResourceId, XGBParams,
)


app = service_app("visualization")
app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/api/datasets")
def list_datasets():
    return api("master", "GET", "/datasets")


@app.post("/api/datasets/import", status_code=201)
def import_dataset(name: Name = Form(), labels: UploadFile = File(), features: list[UploadFile] = File()):
    return api(
        "master", "POST", "/datasets/import", data={"name": name},
        files=upload_parts("labels", [labels]) + upload_parts("features", features),
    )


@app.get("/api/model-defaults")
def model_defaults():
    return ExperimentModels(randomforest=RandomForestParams(), xgb=XGBParams(), catboost=CatBoostParams())


@app.get("/api/experiments")
def list_experiments():
    return api("master", "GET", "/experiments")


@app.post("/api/experiments", status_code=202)
def run_experiment(experiment: Experiment):
    run = api("master", "POST", "/experiments", json=experiment.model_dump())
    return start_experiment(run["id"])


@app.post("/api/experiments/{run_id}/run", status_code=202)
def start_experiment(run_id: ResourceId):
    return api("master", "POST", f"/experiments/{run_id}/run")


@app.get("/api/experiments/{run_id}")
def get_experiment(run_id: ResourceId):
    return api("master", "GET", f"/experiments/{run_id}")


@app.get("/api/experiments/{run_id}/log")
def get_log(run_id: ResourceId):
    return download("master", f"/experiments/{run_id}/log")


@app.get("/api/experiments/{run_id}/artifacts")
def list_artifacts(run_id: ResourceId):
    return api("master", "GET", f"/experiments/{run_id}/artifacts")


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: ResourceId):
    return download("master", f"/artifacts/{artifact_id}")


@app.get("/api/artifacts/{artifact_id}/predictions")
def get_predictions(artifact_id: ResourceId):
    response = request("master", "GET", f"/artifacts/{artifact_id}")
    try:
        reader = csv.DictReader(io.StringIO(response.content.decode("utf-8")))
        if not {"sample_id", "mos", "prediction"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing prediction columns")
        rows = []
        for row in reader:
            mos, prediction = float(row["mos"]), float(row["prediction"])
            if not math.isfinite(mos) or not math.isfinite(prediction):
                raise ValueError("Non-finite MOS or prediction")
            rows.append({
                "sample_id": row["sample_id"], "mos": mos, "prediction": prediction,
                "fold": int(row["fold"]) if "fold" in row else None,
            })
    except (csv.Error, TypeError, ValueError) as exc:
        raise HTTPException(422, f"Invalid prediction CSV: {exc}") from exc
    return rows
