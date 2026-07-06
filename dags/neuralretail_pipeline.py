"""Airflow DAG stub — mirrors the `make pipeline` command.

This file is **documentation-only**. It is **not** meant to be executed
in the single-laptop `docker-compose` dev environment — that runs
`make pipeline` (or `docker compose up mlflow api dashboard` for the
long-running services). The DAG exists to show the production wiring
shape and to drop into an Airflow deployment as-is.

To enable in a real Airflow instance:

1. Provision Airflow 2.x with a DAGs folder that includes this file
   (e.g. mount this repository's ``dags/`` folder into
   ``$AIRFLOW_HOME/dags/``).
2. Ensure the ``neuralretail`` package is on the Airflow worker's
   ``PYTHONPATH`` (install via ``pip install -e .`` or bake into the
   worker image).
3. Trigger manually or wait for the daily schedule.

Each task shells out to a Makefile target so the DAG literally mirrors
the pipeline that actually runs locally — single source of truth, no
duplicated Python logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# Sane defaults; override per-deployment in airflow.cfg.
default_args = {
    "owner": "neuralretail",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="neuralretail_pipeline",
    description=(
        "Daily retail ML refresh: ingest -> features -> train -> drift. "
        "Mirrors `make pipeline`."
    ),
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["neuralretail", "ml", "drift"],
    doc_md=__doc__,
) as dag:
    ingest = BashOperator(
        task_id="ingest_and_clean",
        bash_command="make data",
        doc="Phase 1: ingest raw + clean + Great Expectations validate -> cleaned.parquet",
    )
    features = BashOperator(
        task_id="build_features",
        bash_command="make features",
        doc="Phase 2: RFM + daily-revenue + lag/rolling features",
    )
    train = BashOperator(
        task_id="train_models",
        bash_command="make train",
        doc="Phase 3+4: train Prophet/XGBoost/KMeans/ABC, log to MLflow",
    )
    promote = BashOperator(
        task_id="promote_best",
        bash_command="python -m neuralretail.cli promote",
        doc="Phase 4: alias the best run of each registered model to 'Production'",
    )
    monitor = BashOperator(
        task_id="drift_report",
        bash_command="make monitor",
        doc="Phase 7: Evidently data-drift HTML report (reference vs current)",
    )

    # Linear chain — each step depends on the previous one.
    ingest >> features >> train >> promote >> monitor
