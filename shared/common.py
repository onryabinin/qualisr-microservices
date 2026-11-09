import json
import os
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


config_path = Path(os.getenv("QUALISR_CONFIG", Path(__file__).resolve().parents[1] / "configs/services.json"))
CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
WORKSPACE = Path(CONFIG["workspace"])


def service_app(name, **options):
    app = FastAPI(title=f"QualiSR {name}", **options)

    @app.get("/health")
    def health():
        return {"service": name, "status": "ok"}

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # An overflowing JSON number can appear as infinity in a validation error's input.
        detail = jsonable_encoder(
            exc.errors(), custom_encoder={float: lambda value: value if isfinite(value) else str(value)},
        )
        return JSONResponse({"detail": detail}, status_code=422)

    return app


def now():
    return datetime.now(UTC).isoformat()


def request(service, method, path, **kwargs):
    try:
        response = httpx.request(
            method, CONFIG[service]["url"] + path, timeout=CONFIG["http_timeout"],
            trust_env=False, **kwargs,
        )
    except httpx.RequestError as exc:
        raise HTTPException(503, f"{service} unavailable: {exc}") from exc
    if response.is_error:
        if "application/json" in response.headers.get("content-type", ""):
            detail = response.json().get("detail", response.text)
        else:
            detail = response.text
        raise HTTPException(response.status_code, detail)
    return response


def api(service, method, path, **kwargs):
    return request(service, method, path, **kwargs).json()


def download(service, path):
    response = request(service, "GET", path)
    return Response(
        response.content,
        headers={
            name: response.headers[name]
            for name in ("content-type", "content-disposition")
            if name in response.headers
        },
    )


def upload_parts(field, uploads):
    return [(field, (file.filename, file.file, "text/csv")) for file in uploads]
