"""Thin MLflow wrapper: a local SQLite tracking store in mlruns/ (ignored by Git).

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db     # browse runs at http://localhost:5000

Training works without MLflow installed (the serving image doesn't have it); logging is then skipped.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from src.config import PROJECT_ROOT

TRACKING_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_URI = f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT = "medical-intent-classifier-v2"


def _mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence MLflow's tooling hint on every run
    try:
        import mlflow
    except ImportError:
        return None
    TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI))
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(TRACKING_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


class Run:
    """Logging calls that are no-ops when MLflow isn't available."""

    def __init__(self, mlflow):
        self.mlflow = mlflow

    def params(self, params):
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})

    def metrics(self, metrics, step=None):
        if self.mlflow:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                                    step=step)

    def tags(self, tags):
        if self.mlflow:
            self.mlflow.set_tags(tags)

    def artifacts(self, path, artifact_path=None):
        if self.mlflow and Path(path).exists():
            if Path(path).is_dir():
                self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            else:
                self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    @property
    def run_id(self):
        return self.mlflow.active_run().info.run_id if self.mlflow else None


@contextmanager
def start_run(run_name, tags=None):
    mlflow = None if os.getenv("DISABLE_MLFLOW") == "1" else _mlflow()
    if mlflow is None:
        yield Run(None)
        return
    with mlflow.start_run(run_name=run_name, tags=tags or {}):
        yield Run(mlflow)
