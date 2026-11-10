"""Frozen SigLIP image embeddings with a resumable on-disk cache."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class Encoder:
    def __init__(self, model_id: str):
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def images(self, pil_images) -> np.ndarray:
        inputs = self.processor(images=[im.convert("RGB") for im in pil_images], return_tensors="pt")
        feats = self.model.get_image_features(**inputs)
        feats = getattr(feats, "pooler_output", feats)
        return torch.nn.functional.normalize(feats, dim=-1).numpy()


def embed_cached(encoder: Encoder, ds, indices: list[int], cache_dir: Path, batch: int = 32, chunk: int = 1000) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for c, start in enumerate(range(0, len(indices), chunk)):
        path = cache_dir / f"chunk_{c:04d}_{start}_{len(indices)}.npy"
        if not path.exists():
            ids = indices[start:start + chunk]
            arr = np.concatenate([encoder.images([ds[i]["image"] for i in ids[b:b + batch]]) for b in range(0, len(ids), batch)])
            np.save(path, arr.astype(np.float32))
            print(f"embedded {start + len(ids)}/{len(indices)}", flush=True)
        parts.append(np.load(path))
    return np.concatenate(parts)
