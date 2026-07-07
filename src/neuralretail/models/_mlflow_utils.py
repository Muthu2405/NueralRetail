"""Shared MLflow helpers for all neuralretail model modules.

Centralises tracking-URI setup, experiment creation, and the common
log/promote pattern so each model module only needs to call the helpers.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlflow
import mlflow.sklearn
import pandas as pd

from neuralretail.config import get_settings

logger = logging.getLogger(__name__)

# Map of registered model names — one per logical model in the platform.
REGISTERED_MODEL_NAMES: dict[str, str] = {
    "forecasting": "neuralretail_demand_forecaster",
    "churn": "neuralretail_churn_classifier",
    "segmentation": "neuralretail_customer_segmenter",
    "inventory": "neuralretail_inventory_recommender",
}


def setup_mlflow() -> None:
    """Idempotent: set tracking URI, create experiment, ensure artifact dir."""
    settings = get_settings()
    settings.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(
        "MLflow tracking_uri=%s experiment=%s",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )


@contextmanager
def start_run(run_name: str, tags: Mapping[str, str] | None = None) -> Iterator[Any]:
    """Start an MLflow run; calls setup_mlflow() first."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(dict(tags))
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    """Log a dict of params, stringifying values that aren't numeric/bool."""
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    for k, v in metrics.items():
        if v is None:
            continue
        mlflow.log_metric(k, float(v), step=step)


def log_dataframe_artifact(df: pd.DataFrame, name: str) -> None:
    """Save a small DataFrame as a CSV artifact and return the path."""
    path = Path("mlruns") / "_tmp" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    mlflow.log_artifact(str(path))
    return path


def log_figure(fig: Any, name: str) -> None:
    """Save a matplotlib/plotly figure as a PNG artifact."""
    path = Path("mlruns") / "_tmp" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    except AttributeError:
        # plotly Figure
        fig.write_image(str(path))
    mlflow.log_artifact(str(path))


def log_python_model(
    model: Any,
    artifact_path: str,
    *,
    registered_model_name: str | None = None,
    flavor: str = "auto",
) -> str:
    """Log a model to MLflow.

    flavor:
        - "sklearn" → mlflow.sklearn.log_model (sklearn-compatible estimators)
        - "pyfunc"  → mlflow.pyfunc.log_model with a pickle wrapper
        - "auto"    → sklearn first, pyfunc fallback
    """
    import mlflow

    target = "model"
    if flavor == "sklearn":
        if registered_model_name is None:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        return artifact_path
    if flavor == "pyfunc":
        if registered_model_name is None:
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path, python_model=_PicklePyFunc(model)
            )
        else:
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_PicklePyFunc(model),
                registered_model_name=registered_model_name,
            )
        return artifact_path

    # auto
    try:
        return log_python_model(
            model, artifact_path, registered_model_name=registered_model_name, flavor="sklearn"
        )
    except Exception:
        return log_python_model(
            model, artifact_path, registered_model_name=registered_model_name, flavor="pyfunc"
        )


class _PicklePyFunc(mlflow.pyfunc.PythonModel):
    """Trivial pyfunc wrapper that just pickle-loads the given object."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, context: Any, model_input: list[Any]) -> Any:  # noqa: D401
        return self._model.predict(model_input)


def promote_to_production(run_id: str, model_name: str) -> None:
    """Promote the model version created by ``run_id`` to the 'Production' alias."""
    client = mlflow.tracking.MlflowClient()
    # Get all versions of this registered model, find the one tied to our run.
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            try:
                client.set_registered_model_alias(model_name, "Production", v.version)
                logger.info("Promoted %s v%s to Production", model_name, v.version)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not promote to Production: %s", exc)
            return
    logger.warning("No registered model version found for run_id=%s name=%s", run_id, model_name)


# ---------------------------------------------------------------------------
# Promote best run per registered model
# ---------------------------------------------------------------------------


# Map: logical key -> (registered_model_name, primary_metric, direction)
PROMOTION_CRITERIA: dict[str, tuple[str, str, str]] = {
    "forecasting": (REGISTERED_MODEL_NAMES["forecasting"], "mape", "min"),
    "churn": (REGISTERED_MODEL_NAMES["churn"], "auc_roc", "max"),
    "segmentation": (REGISTERED_MODEL_NAMES["segmentation"], "silhouette", "max"),
    # Inventory's "model quality" is harder to express as a single metric,
    # so we promote the version trained on the most SKUs (deepest history).
    # (dead_stock_pct tracks data coverage, not model skill, and a 0%
    # value just means a tiny test fixture — not a better model.)
    "inventory": (REGISTERED_MODEL_NAMES["inventory"], "n_skus", "max"),
}


def promote_best() -> dict[str, str]:
    """For each registered model, find the best run by primary metric and
    alias it to 'Production'.

    Returns a {model_name: chosen_version} map.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    promoted: dict[str, str] = {}

    for key, (model_name, metric, direction) in PROMOTION_CRITERIA.items():
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not search versions for %s: %s", model_name, exc)
            continue
        if not versions:
            logger.info("No versions registered for %s; skipping", model_name)
            continue

        # Find the best run across all versions of this model
        best_run_id: str | None = None
        best_value: float | None = None
        for v in versions:
            try:
                run = client.get_run(v.run_id)
                value = run.data.metrics.get(metric)
            except Exception:
                continue
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_run_id = v.run_id
                continue
            if direction == "max" and value > best_value:
                best_value = value
                best_run_id = v.run_id
            elif direction == "min" and value < best_value:
                best_value = value
                best_run_id = v.run_id

        if best_run_id is None:
            logger.warning("No run with %s metric for %s; skipping", metric, model_name)
            continue

        # Set the alias on the version tied to the best run
        for v in versions:
            if v.run_id == best_run_id:
                try:
                    client.set_registered_model_alias(model_name, "Production", v.version)
                    promoted[model_name] = v.version
                    logger.info(
                        "Promoted %s v%s to Production (run_id=%s, %s=%.4f)",
                        model_name,
                        v.version,
                        best_run_id,
                        metric,
                        best_value,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Alias set failed for %s: %s", model_name, exc)
                break
    return promoted
