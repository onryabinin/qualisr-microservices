from typing import Annotated, Literal

from fastapi import Path
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ResourceId = Annotated[str, Path(pattern=r"^(qualisr-set120|[0-9a-f]{32})$")]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Group = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
RunStatus = Literal["created", "queued", "running", "succeeded", "failed"]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class DatasetDescription(BaseModel):
    name: Name
    n_samples: int = Field(gt=0)
    n_groups: int = Field(gt=0)
    features: dict[Group, list[str]] = Field(min_length=1)


class RenameDataset(BaseModel):
    name: Name


class ImportCreate(BaseModel):
    name: Name
    type: Literal["builtin", "uploaded"]


class ImportFinish(BaseModel):
    status: Literal["succeeded", "failed"]
    dataset_id: str | None = None
    error: str | None = None


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RandomForestParams(ModelParams):
    n_estimators: int = Field(default=16, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_leaf: int = Field(default=1, ge=1)


class XGBParams(ModelParams):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=6, ge=0)
    learning_rate: float = Field(default=0.3, gt=0, le=1)


class CatBoostParams(ModelParams):
    iterations: int = Field(default=100, ge=1)
    depth: int = Field(default=6, ge=1, le=16)
    learning_rate: float = Field(default=0.03, gt=0, le=1)


class ExperimentModels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    randomforest: RandomForestParams | None = None
    xgb: XGBParams | None = None
    catboost: CatBoostParams | None = None

    @model_validator(mode="after")
    def require_model(self):
        if all(model is None for model in (self.randomforest, self.xgb, self.catboost)):
            raise ValueError("Select at least one regressor")
        return self


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: Annotated[str, StringConstraints(pattern=r"^(qualisr-set120|[0-9a-f]{32})$")]
    features: list[Group] | None = Field(default=None, min_length=1)
    fr_refs: list[Literal["gt", "bicubic", "rlfn", "span"]] = Field(
        default_factory=lambda: ["rlfn"], min_length=1
    )
    mode: Literal["split", "cv"] = "split"
    seed: Seed = 42
    split_seed: Seed = 42
    n_splits: int = Field(default=5, strict=True, ge=2)
    test_size: Literal[0.2] = 0.2
    models: ExperimentModels = Field(
        default_factory=lambda: ExperimentModels(randomforest=RandomForestParams())
    )


class RunUpdate(BaseModel):
    status: RunStatus
    exit_code: int | None = None
    error: str | None = None
    config_path: str | None = None
    log_path: str | None = None
    output_path: str | None = None
    results: dict | None = None
    artifact_paths: list[str] = Field(default_factory=list)
