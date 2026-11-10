"""Train the head, calibrate it, and measure whether its confidence can be trusted, overall and per slice.

Splits: train (fit head), calibration (fit temperature and conformal threshold), test (everything reported).
Usage: python evals/run_eval.py --write
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibcv.calibration import ece, fit_temperature, softmax
from calibcv.conformal import calibrate, prediction_sets
from calibcv.data import SLICE_COLUMNS, sample
from calibcv.encoder import Encoder, embed_cached
from calibcv.model import CalibratedModel

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    enc_id = os.getenv("CC_ENCODER", "google/siglip-base-patch16-224")
    coverage = float(os.getenv("CC_COVERAGE", "0.95"))
    cache = Path(os.getenv("CC_CACHE_DIR", "artifacts/embeddings"))
    ds, tr, ca, te, rows, classes = sample(os.getenv("CC_DATASET", "ashraq/fashion-product-images-small"),
                                           int(os.getenv("CC_IMAGES", "6000")), int(os.getenv("CC_SEED", "11")))
    enc = Encoder(enc_id)
    x = {k: embed_cached(enc, ds, idx, cache / k) for k, idx in (("train", tr), ("calib", ca), ("test", te))}
    y = {k: np.array([rows[i]["label"] for i in idx]) for k, idx in (("train", tr), ("calib", ca), ("test", te))}

    clf = LogisticRegression(C=2.0, max_iter=3000).fit(x["train"], y["train"])
    coef = np.zeros((len(classes), x["train"].shape[1]), np.float32)
    intercept = np.full(len(classes), -1e4, np.float32)
    coef[clf.classes_], intercept[clf.classes_] = clf.coef_, clf.intercept_
    logits = {k: x[k] @ coef.T + intercept for k in x}

    t = fit_temperature(logits["calib"], y["calib"])
    raw_probs, cal_probs = softmax(logits["test"]), softmax(logits["test"] / t)
    ece_raw, _ = ece(raw_probs, y["test"])
    ece_cal, table = ece(cal_probs, y["test"])
    cal_calib = softmax(logits["calib"] / t)
    compare = {}
    for m in ("aps", "lac"):
        qm = calibrate(cal_calib, y["calib"], 1 - coverage, m)
        sm = prediction_sets(softmax(logits["test"] / t), qm, m)
        sz = np.array([len(s) for s in sm])
        compare[m] = (qm, np.mean([yy in s for yy, s in zip(y["test"], sm)]), sz.mean(), np.median(sz), (sz == 1).mean())
    method = os.getenv("CC_METHOD", "lac")
    q = compare[method][0]
    model = CalibratedModel(classes, coef, intercept, t, q, coverage, method)
    model.save(Path(os.getenv("CC_MODEL_DIR", "artifacts/model")))

    sets = prediction_sets(cal_probs, q, method)
    pred = cal_probs.argmax(1)
    covered = np.array([yy in s for yy, s in zip(y["test"], sets)])
    size = np.array([len(s) for s in sets])
    auto = size == 1
    acc = float((pred == y["test"]).mean())

    # Selective prediction with a plain probability threshold, for comparison: the threshold on the
    # calibration split that reaches 99% precision among auto-tagged items.
    cal_conf = softmax(logits["calib"] / t).max(1)
    cal_ok = softmax(logits["calib"] / t).argmax(1) == y["calib"]
    thr = next((c for c in np.sort(cal_conf) if cal_ok[cal_conf >= c].mean() >= 0.99), 1.0)
    thr_auto = cal_probs.max(1) >= thr

    lines = ["# Eval results", "",
             f"Encoder `{enc_id}` (frozen) + logistic-regression head, {len(classes)} article types. "
             f"Splits: {len(tr)} train, {len(ca)} calibration, {len(te)} test. Target coverage {coverage:.0%}.", "",
             "| Metric | Value |", "| --- | --- |",
             f"| Test accuracy (top-1) | {acc:.1%} |",
             f"| Temperature | {t:.2f} |",
             f"| ECE before / after temperature scaling | {ece_raw:.3f} / {ece_cal:.3f} |",
             f"| Conformal method served | {method.upper()} |",
             f"| Conformal set coverage on test | {covered.mean():.1%} (target {coverage:.0%}) |",
             f"| Mean set size | {size.mean():.2f} |",
             f"| Auto-tagged (set of one) | {auto.mean():.1%} of images, {(pred[auto] == y['test'][auto]).mean():.1%} correct |",
             f"| Sent to review | {(~auto).mean():.1%} of images, true label in the set {covered[~auto].mean():.1%} of the time |",
             f"| Probability threshold for 99% precision on calibration ({thr:.3f}) | auto-tags {thr_auto.mean():.1%} of test, {(pred[thr_auto] == y['test'][thr_auto]).mean():.1%} correct |", "",
             "## Conformal score functions compared (same calibration split, test set)", "",
             "| Method | Threshold q | Coverage | Mean set size | Median set size | Sets of one |", "| --- | --- | --- | --- | --- | --- |",
             *[f"| {m.upper()} | {v[0]:.4f} | {v[1]:.1%} | {v[2]:.2f} | {v[3]:.0f} | {v[4]:.1%} |" for m, v in compare.items()], "",
             "## Per slice (test set)", "",
             "| Slice | Images | Accuracy | Set coverage | Auto-tag rate | Auto-tag accuracy |", "| --- | --- | --- | --- | --- | --- |"]
    flagged = []
    for col in SLICE_COLUMNS:
        groups = defaultdict(list)
        for k, i in enumerate(te):
            groups[rows[i][col]].append(k)
        for value, ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(ks) < 30:
                continue  # too few images for a stable number
            ks = np.array(ks)
            cov = covered[ks].mean()
            a = auto[ks]
            auto_acc = (pred[ks][a] == y["test"][ks][a]).mean() if a.any() else float("nan")
            lines.append(f"| {col} = {value} | {len(ks)} | {(pred[ks] == y['test'][ks]).mean():.1%} | {cov:.1%} | {a.mean():.1%} | {auto_acc:.1%} |")
            if cov < coverage - 0.03:
                flagged.append(f"{col} = {value} ({cov:.1%})")
    lines += ["", "Slices more than 3 points under the coverage target: " + (", ".join(flagged) if flagged else "none") + ".",
              "", "## Reliability after temperature scaling (test set)", "", "| Confidence bin | Images | Mean confidence | Accuracy |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['bin']} | {r['count']} | {r['confidence']:.3f} | {r['accuracy']:.3f} |" for r in table]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.write:
        (ROOT / "evals/RESULTS.md").write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
