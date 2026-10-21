"""Guards for the deployment configuration (a mismatch here once broke a Render deploy)."""

import re

import yaml

from src.config import resolve


def render_service():
    return yaml.safe_load(resolve("render.yaml").read_text(encoding="utf-8"))["services"][0]


def render_env():
    return {e["key"]: str(e["value"]) for e in render_service()["envVars"]}


def test_render_builds_from_a_pinned_hub_revision():
    env = render_env()
    assert env["MODEL_SOURCE"] == "hub"
    assert re.fullmatch(r"[0-9a-f]{40}", env["HF_MODEL_REVISION"]), "pin a full commit hash, never a branch"
    assert env["HF_MODEL_REPO"].count("/") == 1


def test_render_deploys_only_after_ci_passes():
    service = render_service()
    assert service["plan"] == "free"
    assert service["autoDeployTrigger"] == "checksPass"
    assert service["healthCheckPath"] == "/health"


def test_dockerfile_last_stage_is_the_served_image():
    """Render can't choose a build target, so the deployed stage must be the last one."""
    stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", resolve("Dockerfile").read_text(encoding="utf-8"), re.MULTILINE)
    assert stages[-1] == "serve"


def test_served_image_matches_the_deployed_model_type():
    """The serve stage's backend must match the artifacts render.yaml points at (v2 = transformer)."""
    dockerfile = resolve("Dockerfile").read_text(encoding="utf-8")
    serve_stage = dockerfile[dockerfile.rindex("AS serve"):]
    assert "MODEL_BACKEND=transformer-onnx" in serve_stage
    assert "INTENTS_FILE=data/v2/intents.json" in serve_stage
    assert "COPY web web" in serve_stage
