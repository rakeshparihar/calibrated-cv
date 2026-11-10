"""Split conformal prediction: two score functions.

LAC (least ambiguous classifier): score = 1 - probability of the true class. A class joins the set
when its probability is at least 1 - q. Gives the smallest sets on average; set size does not adapt
much to how hard an image is.

APS (adaptive prediction sets): score = total probability of the classes ranked above and including
the true class; classes join in order until their total reaches q. Sets adapt to difficulty, but
without APS's randomised tie-break (omitted here) a confident model drives q close to 1, and every
set then has to collect almost all probability mass: sets become very large. The eval measures this.

Calibration for both: the ceil((n + 1)(1 - alpha)) / n quantile q of the calibration scores.
Guarantee for both: if calibration and new images are exchangeable, the set contains the true class
with probability at least 1 - alpha, averaged over images (marginal, not per slice).
"""
import math

import numpy as np

METHODS = ("lac", "aps")


def aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank_of_true = np.argmax(order == y[:, None], axis=1)
    return cum[np.arange(len(y)), rank_of_true]


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(y)), y]


def calibrate(probs: np.ndarray, y: np.ndarray, alpha: float, method: str = "lac") -> float:
    n = len(y)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    scores = lac_scores(probs, y) if method == "lac" else aps_scores(probs, y)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, q: float, method: str = "lac") -> list[list[int]]:
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    if method == "lac":
        # never return an empty set: keep the top class even when it is below the threshold
        sizes = np.maximum((sorted_p >= 1.0 - q).sum(axis=1), 1)
    else:
        sizes = (np.cumsum(sorted_p, axis=1) < q).sum(axis=1) + 1  # include the class that crosses q
    return [order[i, : sizes[i]].tolist() for i in range(len(probs))]
