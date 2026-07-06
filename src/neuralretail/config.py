"""Application configuration loaded from environment / .env.

All other modules should import paths and runtime settings from here,
not from the environment directly. This keeps secrets, paths, and feature
flags in a single typed source of truth.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the neuralretail platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NEURALRETAIL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = Field(default=Path("./data"))
    raw_dir: Path = Field(default=Path("./data/raw"))
    processed_dir: Path = Field(default=Path("./data/processed"))
    models_dir: Path = Field(default=Path("./models"))
    report_dir: Path = Field(default=Path("./report"))

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="sqlite:///./mlruns/mlflow.db")
    mlflow_artifact_root: Path = Field(default=Path("./mlruns/artifacts"))
    mlflow_experiment_name: str = Field(default="neuralretail")

    # --- API ---
    api_key: str = Field(default="change-me-in-prod")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Dashboard ---
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8501)

    # --- Feature flags ---
    enable_lstm: bool = Field(default=False)
    enable_lightgbm: bool = Field(default=False)
    enable_dbscan: bool = Field(default=False)

    # --- Monitoring / drift ---
    # Fraction of the cleaned data used as the drift "reference" window
    # (oldest slice); the remainder is the "current" window. Default 0.7
    # mirrors a typical 70/30 reference/current split.
    drift_reference_fraction: float = Field(default=0.7)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    def ensure_directories(self) -> None:
        """Create the standard project directories if they don't exist."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.report_dir,
            self.mlflow_artifact_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
