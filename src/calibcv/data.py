"""Sample the dataset, split train / calibration / test, and keep slice columns for evaluation."""
import random
from collections import Counter

from datasets import load_dataset

SLICE_COLUMNS = ("gender", "season", "masterCategory")


def sample(dataset: str, n: int, seed: int, top_types: int = 25, fractions=(0.6, 0.2, 0.2)):
    ds = load_dataset(dataset, split="train")
    meta = ds.remove_columns(["image"]).to_list()
    common = {t for t, _ in Counter(r["articleType"] for r in meta).most_common(top_types)}
    keep = [i for i, r in enumerate(meta) if r["articleType"] in common]
    rng = random.Random(seed)
    rng.shuffle(keep)
    keep = keep[:n]
    a, b = int(fractions[0] * len(keep)), int((fractions[0] + fractions[1]) * len(keep))
    classes = sorted(common)
    rows = {i: {"label": classes.index(meta[i]["articleType"]), **{c: meta[i][c] or "unknown" for c in SLICE_COLUMNS}} for i in keep}
    return ds, keep[:a], keep[a:b], keep[b:], rows, classes
