import csv
import io
import logging
import math
import re
from contextlib import ExitStack, asynccontextmanager, contextmanager
from itertools import islice
from pathlib import Path

import pandas as pd
from fastapi import File, Form, HTTPException, Query, UploadFile

from shared.common import CONFIG, api, request, service_app, upload_parts
from shared.contracts import DatasetDescription, Group, Name, ResourceId


LOGGER = logging.getLogger("uvicorn.error")
SETTINGS = CONFIG["collector"]
BUILTIN_ID = "qualisr-set120"
METHOD_ALIASES = {
    "pasd": "PASD", "supir": "SUPIR", "realesrgan": "RealESRGAN",
    "real_esrgan": "RealESRGAN", "real-esrgan": "RealESRGAN",
}
FEATURE_METADATA = {
    "sample_id", "dataset", "test_case", "correlation_group", "score_type", "rel_path",
    "sr_method", "sr_filename", "sr_path", "gt_path", "lr_path", "set_type",
}
FEATURE_FILE = re.compile(r"([a-z][a-z0-9_]{0,63})\.csv")


@contextmanager
def csv_rows(upload):
    upload.file.seek(0)
    text = io.TextIOWrapper(upload.file, encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(text, strict=True)
        fields = reader.fieldnames or []
        if not fields or any(not name.strip() for name in fields) or len(set(fields)) != len(fields):
            raise ValueError(f"{upload.filename}: empty or duplicate CSV headers")
        if any("\0" in name for name in fields):
            raise ValueError(f"{upload.filename}: NUL byte in CSV header")
        yield reader, fields
    finally:
        text.detach()
        upload.file.seek(0)


def check_row(reader, row, filename):
    if None in row or None in row.values():
        raise ValueError(f"{filename}:{reader.line_num}: wrong number of columns")
    if any("\0" in value for value in row.values()):
        raise ValueError(f"{filename}:{reader.line_num}: NUL byte in CSV")


def finite_number(value):
    try:
        return value.isascii() and "_" not in value and math.isfinite(float(value))
    except ValueError:
        return False


def label_keys(upload):
    required = {"method", "test_case", "score"}
    with csv_rows(upload) as (reader, columns):
        if not required <= set(columns):
            raise ValueError("labels.csv requires method, test_case and score")
        for row in reader:
            check_row(reader, row, "labels.csv")
            if not finite_number(row["score"]):
                raise ValueError(f"labels.csv:{reader.line_num}: score must be finite and numeric")

    # Match the type inference in the standalone loader before formatting IDs.
    labels = pd.read_csv(upload.file, usecols=list(required))
    upload.file.seek(0)
    keys = set()
    groups = set()
    for number, row in enumerate(labels.to_dict("records"), start=1):
        method, case = str(row["method"]).strip(), str(row["test_case"]).strip()
        if pd.isna(row["method"]) or pd.isna(row["test_case"]) or not method or not case:
            raise ValueError(f"labels.csv sample {number}: missing method or test_case")
        if any(char in method + case for char in "/\\\r\n"):
            raise ValueError(f"labels.csv sample {number}: invalid method or test_case")
        method = METHOD_ALIASES.get(method.casefold(), method)
        key = f"QualiSR-Set120/{method}/{case}.npy.gz"
        if key in keys:
            raise ValueError(f"labels.csv: duplicate sample {key}")
        keys.add(key)
        groups.add(case)
    if not keys:
        raise ValueError("labels.csv is empty")
    return keys, len(groups)


def feature_columns(upload, keys):
    found = set()
    with csv_rows(upload) as (reader, columns):
        if "sample_id" not in columns or "score" in columns:
            raise ValueError(f"{upload.filename}: requires sample_id and must omit the target score")
        numeric_columns = [column for column in columns if column not in FEATURE_METADATA]
        if not numeric_columns:
            raise ValueError(f"{upload.filename}: requires a feature column")
        for row in reader:
            check_row(reader, row, upload.filename)
            key = row["sample_id"]
            if key not in keys or key in found:
                raise ValueError(f"{upload.filename}: unknown or duplicate sample_id {key!r}")
            found.add(key)
            for column in numeric_columns:
                value = row[column]
                if value not in {"", "NaN", "nan"} and not finite_number(value):
                    raise ValueError(f"{upload.filename}:{reader.line_num}: invalid number in {column}")
    if found != keys:
        raise ValueError(f"{upload.filename}: missing {len(keys - found)} sample IDs from labels.csv")
    return numeric_columns


def describe_dataset(name, labels, features):
    if labels.filename != "labels.csv" or not features:
        raise ValueError("Upload labels.csv and at least one feature CSV")
    total_bytes = sum(file.file.seek(0, io.SEEK_END) for file in [labels, *features])
    if total_bytes > SETTINGS["max_upload_bytes"]:
        raise HTTPException(413, f"Dataset exceeds {SETTINGS['max_upload_bytes']} bytes")
    keys, n_groups = label_keys(labels)
    groups = {}
    for upload in features:
        match = FEATURE_FILE.fullmatch(upload.filename or "")
        if not match or match[1] == "labels" or match[1] in groups:
            raise ValueError(f"Invalid or duplicate feature filename {upload.filename!r}")
        groups[match[1]] = feature_columns(upload, keys)
    return DatasetDescription(name=name, n_samples=len(keys), n_groups=n_groups, features=groups)


def register(name, labels, features, kind):
    entry = api("storage", "POST", "/imports", json={"name": name, "type": kind})
    import_path = f"/imports/{entry['id']}"
    try:
        description = describe_dataset(name, labels, features)
        dataset = api(
            "storage", "POST", "/datasets",
            data={"description": description.model_dump_json(), "import_id": entry["id"]},
            files=upload_parts("files", [labels, *features]),
        )
    except (ValueError, csv.Error, HTTPException) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        LOGGER.warning("Import %s failed: %s", entry["id"], detail)
        try:
            api("storage", "PATCH", import_path, json={"status": "failed", "error": str(detail)})
        except HTTPException:
            LOGGER.exception("Could not record failure of import %s", entry["id"])
        status = exc.status_code if isinstance(exc, HTTPException) else 422
        raise HTTPException(status, {"import_id": entry["id"], "error": detail}) from exc
    LOGGER.info("Imported %s as dataset %s (%s samples)", entry["id"], dataset["id"], dataset["n_samples"])
    return {"dataset": dataset, "import_id": entry["id"]}


@asynccontextmanager
async def lifespan(_app):
    try:
        api("storage", "GET", f"/datasets/{BUILTIN_ID}")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        root = Path(SETTINGS["builtin_root"])
        with ExitStack() as stack:
            def upload(path):
                return UploadFile(file=stack.enter_context(path.open("rb")), filename=path.name)

            paths = sorted((root / "features").glob("*.csv"))
            paths.extend(sorted((root / "features/pca").glob("*.csv")))
            register("QualiSR-Set120", upload(root / "labels.csv"), [upload(path) for path in paths], "builtin")
    yield


app = service_app("collector", lifespan=lifespan)


@app.post("/imports", status_code=201)
def import_upload(name: Name = Form(), labels: UploadFile = File(), features: list[UploadFile] = File()):
    return register(name, labels, features, "uploaded")


@app.get("/imports")
def list_imports():
    return api("storage", "GET", "/imports")


@app.get("/imports/{import_id}")
def get_import(import_id: ResourceId):
    return api("storage", "GET", f"/imports/{import_id}")


@app.get("/datasets/{dataset_id}/batches")
def get_batch(
    dataset_id: ResourceId,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    feature: Group | None = None,
):
    dataset = api("storage", "GET", f"/datasets/{dataset_id}")
    group = feature or "labels"
    total = dataset["n_samples"]
    if group != "labels" and group not in dataset["features"]:
        raise HTTPException(404, f"CSV group {group} not found")
    rows = []
    if offset < total:
        response = request("storage", "GET", f"/datasets/{dataset_id}/files/{group}")
        content = io.StringIO(response.content.decode("utf-8-sig"), newline="")
        rows = list(islice(csv.DictReader(content), offset, offset + limit))
    LOGGER.info("Batch %s/%s offset=%s limit=%s returned=%s", dataset_id, group, offset, limit, len(rows))
    return {"offset": offset, "limit": limit, "total": total, "rows": rows}
