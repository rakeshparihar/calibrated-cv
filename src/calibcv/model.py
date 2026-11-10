"""The served model: a linear head on frozen SigLIP embeddings, a temperature, and a conformal threshold."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import softmax
from .conformal import prediction_sets


@dataclass
class CalibratedModel:
    classes: list[str]
    coef: np.ndarray
    intercept: np.ndarray
    temperature: float
    q: float
    coverage: float
    method: str = "lac"

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef.T + self.intercept

    def predict(self, x: np.ndarray) -> list[dict]:
        probs = softmax(self.logits(x) / self.temperature)
        out = []
        for p, s in zip(probs, prediction_sets(probs, self.q, self.method)):
            labels = [self.classes[i] for i in s]
            out.append({
                "label": self.classes[int(p.argmax())],
                "probability": round(float(p.max()), 4),
                "prediction_set": labels,
                # The contract: a single-label set is safe to auto-tag; anything larger goes to a person,
                # who chooses from the set instead of the full class list.
                "decision": "auto_tag" if len(labels) == 1 else "review",
            })
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "head.npz", coef=self.coef, intercept=self.intercept)
        (directory / "meta.json").write_text(json.dumps({"classes": self.classes, "temperature": self.temperature,
                                                         "q": self.q, "coverage": self.coverage,
                                                         "method": self.method}))

    @classmethod
    def load(cls, directory: Path) -> "CalibratedModel":
        a = np.load(directory / "head.npz")
        m = json.loads((directory / "meta.json").read_text())
        return cls(m["classes"], a["coef"], a["intercept"], m["temperature"], m["q"], m["coverage"], m.get("method", "lac"))
