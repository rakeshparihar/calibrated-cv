"""Temperature scaling and calibration error.

Temperature scaling divides the logits by one fitted number T. It does not
change which class wins, so accuracy is unchanged; it only makes the
probabilities match how often the model is actually right.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    p = softmax(logits / t)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    return float(minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20), method="bounded").x)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error: average gap between confidence and accuracy, weighted by bin size."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            total += m.mean() * abs(acc - c)
            table.append({"bin": f"{lo:.2f}-{hi:.2f}", "count": int(m.sum()), "confidence": c, "accuracy": acc})
    return float(total), table
