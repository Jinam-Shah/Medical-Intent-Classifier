import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.config import PROJECT_ROOT
from src.export_onnx import compare, export
from src.onnx_predict import OnnxIntentClassifier
from tests.conftest import LABELS

TEXTS = ["I have a fever", "hello", "a cough", "", "completely unseen words here",
         " ".join(["fever"] * 30)]  # longer than max_seq_length, so it is truncated


@pytest.fixture(scope="module")
def onnx_model(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx")
    ckpt = out / "model.pt"
    ckpt.write_bytes(tiny_checkpoint.read_bytes())
    torch_clf = export(ckpt, out / "model.onnx")
    return torch_clf, out / "model.onnx"


def test_onnx_matches_pytorch(onnx_model):
    torch_clf, path = onnx_model
    report = compare(torch_clf, OnnxIntentClassifier(path), TEXTS)
    assert report["argmax_agreement"] == 1.0
    assert report["max_abs_prob_diff"] < 1e-5


def test_onnx_classifier_interface(onnx_model):
    _, path = onnx_model
    clf = OnnxIntentClassifier(path, num_threads=1)
    assert clf.labels == LABELS and clf.backend == "onnx"
    probs = clf.predict_proba(TEXTS)
    assert probs.shape == (len(TEXTS), len(LABELS))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    pred = clf.predict("I have a fever")
    assert pred["intent"] in LABELS and len(pred["top_k"]) == 3


def test_api_onnx_backend(onnx_model):
    _, path = onnx_model
    with TestClient(create_app(model_path=path, backend="onnx", confidence_threshold=0.0)) as c:
        assert c.get("/model-info").json()["backend"] == "onnx"
        body = c.post("/predict", json={"text": "I have a fever"}).json()
        assert body["intent"] in LABELS
        batch = c.post("/predict/batch", json={"texts": ["hello", "a cough"]}).json()["results"]
        assert len(batch) == 2


def test_onnx_serving_path_never_imports_torch(onnx_model):
    """The slim Docker image has no torch; the API on the ONNX backend must work without it."""
    _, path = onnx_model
    script = (
        "import sys\n"
        "class BlockTorch:\n"  # behaves as if torch were not installed
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked in this test')\n"
        "sys.meta_path.insert(0, BlockTorch())\n"
        "from fastapi.testclient import TestClient\n"
        "from api.main import create_app\n"
        f"app = create_app(model_path={str(path)!r}, backend='onnx')\n"
        "with TestClient(app) as c:\n"
        "    assert c.post('/predict', json={'text': 'I have a fever'}).status_code == 200\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
