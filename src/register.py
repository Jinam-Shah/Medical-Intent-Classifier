"""Log the chosen v2 model (ONNX) to MLflow with its metrics and register it in the model registry.

    python -m src.register
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db    # Models tab -> medical-intent-classifier
"""

import json

from src.config import resolve
from src.tracking import start_run

MODEL_NAME = "medical-intent-classifier"
CHOSEN = {"name": "minilm_v2", "onnx_dir": "outputs/models/minilm_v2/onnx_fp32"}


def main():
    import mlflow
    import onnx

    onnx_dir = resolve(CHOSEN["onnx_dir"])
    reports = resolve(f"reports/v2/{CHOSEN['name']}")
    val = json.loads((reports / "val_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((reports / "test_metrics.json").read_text(encoding="utf-8"))
    serving = json.loads((onnx_dir / "serving.json").read_text(encoding="utf-8"))

    with start_run(f"{CHOSEN['name']}-registered", tags={"model": CHOSEN["name"], "stage": "chosen"}) as run:
        run.params({"model": CHOSEN["name"], "format": "onnx-fp32", **serving})
        run.metrics({"val_accuracy": val["classification"]["accuracy"],
                     "val_f1_macro": val["classification"]["f1_macro"],
                     "test_accuracy": test["classification"]["accuracy"],
                     "test_f1_macro": test["classification"]["f1_macro"],
                     "test_end_to_end_accuracy": test["end_to_end"]["end_to_end_accuracy"],
                     "test_clinc_accuracy": test["by_source"]["clinc (independent)"]["classification_accuracy"]})
        for name in ("model_meta.json", "serving.json", "tokenizer.json"):
            run.artifacts(onnx_dir / name, artifact_path="serving")
        run.artifacts(reports, artifact_path="reports")
        info = mlflow.onnx.log_model(onnx.load(str(onnx_dir / "model.onnx")), name="model",
                                     registered_model_name=MODEL_NAME)
        print(f"run {run.run_id}: registered {MODEL_NAME} version {info.registered_model_version}")


if __name__ == "__main__":
    main()
