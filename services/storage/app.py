import json
import shutil
import sqlite3
from contextlib import asynccontextmanager, closing, contextmanager
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from shared.common import WORKSPACE, now, service_app
from shared.contracts import (
    DatasetDescription, Experiment, ImportCreate, ImportFinish, RenameDataset,
    ResourceId, RunStatus, RunUpdate,
)


DATASETS = WORKSPACE / "datasets"
RUNS = WORKSPACE / "runs"
DATABASE = WORKSPACE / "metadata" / "storage.sqlite3"
TRANSITIONS = {
    "created": {"queued"},
    "queued": {"running"},
    "running": {"succeeded", "failed"},
    "succeeded": set(),
    "failed": set(),
}


@contextmanager
def connection():
    with closing(sqlite3.connect(DATABASE)) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        yield db


@asynccontextmanager
async def lifespan(_app):
    for directory in (DATASETS, RUNS, DATABASE.parent):
        directory.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS imports (
                id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                metadata TEXT NOT NULL
            );
        """)
    yield


app = service_app("storage", lifespan=lifespan)


def record(db, table, item_id):
    row = db.execute(f"SELECT metadata FROM {table} WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"{table}/{item_id} not found")
    return json.loads(row[0])


def records(db, query, params=()):
    return [json.loads(row[0]) for row in db.execute(query, params)]


def describe_file(path):
    with path.open("rb") as stream:
        return {"size": path.stat().st_size, "sha256": file_digest(stream, "sha256").hexdigest()}


def finish_import(db, import_id, update):
    entry = record(db, "imports", import_id)
    if entry["status"] != "running":
        raise HTTPException(409, "Import has already finished")
    entry.update(update.model_dump(), finished_at=now())
    db.execute("UPDATE imports SET metadata = ? WHERE id = ?", (json.dumps(entry), import_id))
    return entry


@app.post("/imports", status_code=201)
def create_import(body: ImportCreate):
    entry = {**body.model_dump(), "id": uuid4().hex, "status": "running", "created_at": now()}
    with connection() as db:
        db.execute("INSERT INTO imports VALUES (?, ?)", (entry["id"], json.dumps(entry)))
    return entry


@app.get("/imports")
def list_imports():
    with connection() as db:
        return records(db, "SELECT metadata FROM imports ORDER BY rowid")


@app.get("/imports/{import_id}")
def get_import(import_id: ResourceId):
    with connection() as db:
        return record(db, "imports", import_id)


@app.patch("/imports/{import_id}")
def update_import(import_id: ResourceId, update: ImportFinish):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        return finish_import(db, import_id, update)


@app.post("/datasets", status_code=201)
def create_dataset(description: str = Form(), import_id: str = Form(), files: list[UploadFile] = File()):
    try:
        info = DatasetDescription.model_validate_json(description)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    expected_files = {"labels.csv": "labels.csv"}
    expected_files.update({f"{group}.csv": f"features/{group}.csv" for group in info.features})
    names = [file.filename for file in files]
    if len(names) != len(expected_files) or set(names) != set(expected_files):
        raise HTTPException(422, "Uploaded filenames do not match the dataset description")

    with connection() as db, TemporaryDirectory(dir=DATASETS) as staging:
        entry = record(db, "imports", import_id)
        if entry["status"] != "running":
            raise HTTPException(409, "Import has already finished")
        dataset_id = "qualisr-set120" if entry["type"] == "builtin" else uuid4().hex
        source = Path(staging)
        (source / "features").mkdir()
        manifest = {}
        for upload in files:
            relative = expected_files[upload.filename]
            path = source / relative
            with path.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            manifest[relative] = describe_file(path)
        dataset = {
            **info.model_dump(), "id": dataset_id, "type": entry["type"], "files": manifest,
            "created_at": now(), "updated_at": now(),
        }
        db.execute("BEGIN IMMEDIATE")
        target = DATASETS / dataset_id
        if target.exists():
            raise HTTPException(409, f"Dataset {dataset_id} already exists")
        db.execute("INSERT INTO datasets VALUES (?, ?)", (dataset_id, json.dumps(dataset)))
        finish_import(db, import_id, ImportFinish(status="succeeded", dataset_id=dataset_id))
        source.rename(target)
        try:
            db.commit()
        except sqlite3.Error:
            target.rename(source)
            raise
    return dataset


@app.get("/datasets")
def list_datasets():
    with connection() as db:
        return records(db, "SELECT metadata FROM datasets ORDER BY rowid")


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: ResourceId):
    with connection() as db:
        return record(db, "datasets", dataset_id)


@app.patch("/datasets/{dataset_id}")
def rename_dataset(dataset_id: ResourceId, body: RenameDataset):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        dataset = record(db, "datasets", dataset_id)
        dataset.update(name=body.name, updated_at=now())
        db.execute("UPDATE datasets SET metadata = ? WHERE id = ?", (json.dumps(dataset), dataset_id))
    return dataset


def delete_files_and_record(db, table, item_id, directory):
    with TemporaryDirectory(dir=directory.parent) as staging:
        target = Path(staging) / item_id
        db.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
        if directory.exists():
            directory.rename(target)
        try:
            db.commit()
        except sqlite3.Error:
            if target.exists():
                target.rename(directory)
            raise


@app.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: ResourceId):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        dataset = record(db, "datasets", dataset_id)
        if dataset["type"] == "builtin":
            raise HTTPException(409, "The built-in dataset cannot be deleted")
        if db.execute("SELECT 1 FROM runs WHERE dataset_id = ?", (dataset_id,)).fetchone():
            raise HTTPException(409, "Dataset has experiments")
        delete_files_and_record(db, "datasets", dataset_id, DATASETS / dataset_id)


@app.get("/datasets/{dataset_id}/files/{group}")
def get_dataset_file(dataset_id: ResourceId, group: str):
    dataset = get_dataset(dataset_id)
    relative = "labels.csv" if group == "labels" else f"features/{group}.csv"
    if relative not in dataset["files"]:
        raise HTTPException(404, f"CSV group {group} not found")
    return FileResponse(DATASETS / dataset_id / relative, media_type="text/csv")


@app.post("/runs", status_code=201)
def create_run(experiment: Experiment):
    run = {
        **experiment.model_dump(), "id": uuid4().hex, "status": "created",
        "created_at": now(), "queued_at": None, "started_at": None, "finished_at": None,
        "exit_code": None, "error": None, "results": None, "config_path": None,
        "log_path": None, "output_path": None,
    }
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        record(db, "datasets", experiment.dataset_id)
        db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            (run["id"], experiment.dataset_id, run["status"], json.dumps(run)),
        )
    return run


@app.get("/runs")
def list_runs(status: RunStatus | None = None):
    with connection() as db:
        if status == "queued":
            return records(db, """
                SELECT metadata FROM runs WHERE status = 'queued'
                ORDER BY json_extract(metadata, '$.queued_at'), rowid
            """)
        if status:
            return records(db, "SELECT metadata FROM runs WHERE status = ? ORDER BY rowid", (status,))
        return records(db, "SELECT metadata FROM runs ORDER BY rowid")


@app.get("/runs/{run_id}")
def get_run(run_id: ResourceId):
    with connection() as db:
        return record(db, "runs", run_id)


def register_artifacts(db, run_id, paths):
    base = RUNS / run_id
    for relative in sorted(set(paths)):
        path = (base / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(base) or not path.is_file():
            raise HTTPException(422, f"Invalid artifact {relative}")
        artifact = {
            **describe_file(path), "id": uuid4().hex, "run_id": run_id,
            "path": relative, "name": path.name,
        }
        db.execute("INSERT INTO artifacts VALUES (?, ?, ?)", (
            artifact["id"], run_id, json.dumps(artifact),
        ))


@app.patch("/runs/{run_id}")
def update_run(run_id: ResourceId, update: RunUpdate):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        run = record(db, "runs", run_id)
        if update.status not in TRANSITIONS[run["status"]]:
            raise HTTPException(409, f"Cannot change run from {run['status']} to {update.status}")
        fields = update.model_dump(exclude_unset=True, exclude={"artifact_paths"})
        if update.status == "queued":
            run["queued_at"] = now()
        elif update.status == "running":
            run["started_at"] = now()
        else:
            run["finished_at"] = now()
            register_artifacts(db, run_id, update.artifact_paths)
        run.update(fields)
        db.execute(
            "UPDATE runs SET status = ?, metadata = ? WHERE id = ?",
            (update.status, json.dumps(run), run_id),
        )
    return run


@app.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: ResourceId):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        run = record(db, "runs", run_id)
        if run["status"] in {"queued", "running"}:
            raise HTTPException(409, "Experiment is active")
        delete_files_and_record(db, "runs", run_id, RUNS / run_id)


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: ResourceId):
    with connection() as db:
        record(db, "runs", run_id)
        return records(db, "SELECT metadata FROM artifacts WHERE run_id = ? ORDER BY rowid", (run_id,))


@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: ResourceId):
    with connection() as db:
        artifact = record(db, "artifacts", artifact_id)
    return FileResponse(RUNS / artifact["run_id"] / artifact["path"], filename=artifact["name"])
