import numpy as np
from fastapi.testclient import TestClient

from calibcv.api import create_app
from calibcv.calibration import ece, fit_temperature, softmax
from calibcv.conformal import aps_scores, calibrate, prediction_sets
from calibcv.model import CalibratedModel


def synthetic(n=4000, k=5, seed=0, sharpness=3.0):
    """Logits from a known model: true class gets a boost, then everything is made overconfident."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5
    return logits * sharpness, y


def test_temperature_reduces_ece_and_keeps_argmax():
    logits, y = synthetic()
    t = fit_temperature(logits, y)
    assert t > 1.5  # overconfident logits need softening
    before, _ = ece(softmax(logits), y)
    after, _ = ece(softmax(logits / t), y)
    assert after < before / 2
    assert (softmax(logits).argmax(1) == softmax(logits / t).argmax(1)).all()


def test_conformal_sets_reach_target_coverage_on_new_data():
    logits, y = synthetic(n=6000, seed=1)
    probs = softmax(logits / fit_temperature(logits, y))
    for method in ("lac", "aps"):
        q = calibrate(probs[:3000], y[:3000], alpha=0.1, method=method)
        sets = prediction_sets(probs[3000:], q, method)
        coverage = np.mean([yy in s for yy, s in zip(y[3000:], sets)])
        assert coverage >= 0.88, method  # 0.90 target, minus sampling noise


def test_aps_score_is_mass_up_to_true_class():
    probs = np.array([[0.6, 0.3, 0.1]])
    assert np.allclose(aps_scores(probs, np.array([1])), [0.9])
    assert prediction_sets(probs, 0.85, "aps") == [[0, 1]]
    assert prediction_sets(probs, 0.5, "aps") == [[0]]


def test_lac_set_is_classes_above_threshold_never_empty():
    probs = np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
    assert prediction_sets(probs, 0.75, "lac") == [[0, 1], [0, 1, 2]]
    assert prediction_sets(probs, 0.1, "lac") == [[0], [0]]


def test_model_decision_and_round_trip(tmp_path):
    coef = np.eye(3, dtype=np.float32) * 10
    model = CalibratedModel(["a", "b", "c"], coef, np.zeros(3, np.float32), 1.0, 0.9, 0.95)
    sure, unsure = model.predict(np.array([[1.0, 0, 0], [0.5, 0.5, 0]], dtype=np.float32))
    assert sure["decision"] == "auto_tag" and sure["prediction_set"] == ["a"]
    assert unsure["decision"] == "review" and set(unsure["prediction_set"]) == {"a", "b"}
    model.save(tmp_path)
    assert CalibratedModel.load(tmp_path).predict(np.array([[1.0, 0, 0]], dtype=np.float32))[0]["label"] == "a"


def test_api_classifies_an_upload():
    class FakeEncoder:
        def images(self, imgs):
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    model = CalibratedModel(["a", "b", "c"], np.eye(3, dtype=np.float32) * 10, np.zeros(3, np.float32), 1.0, 0.9, 0.95)
    client = TestClient(create_app(model, FakeEncoder()))
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    body = client.post("/classify", files={"file": ("x.png", buf.getvalue())}).json()
    assert body["label"] == "b" and body["decision"] == "auto_tag" and body["coverage_target"] == 0.95
    assert client.post("/classify", files={"file": ("x.png", b"nope")}).status_code == 400
