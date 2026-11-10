"""HTTP API: upload a product photo, get a label, a calibrated probability, a prediction set and a decision."""
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .model import CalibratedModel


def create_app(model: CalibratedModel | None = None, encoder=None) -> FastAPI:
    if model is None:
        model = CalibratedModel.load(Path(os.getenv("CC_MODEL_DIR", "artifacts/model")))
    if encoder is None:
        from .encoder import Encoder

        encoder = Encoder(os.getenv("CC_ENCODER", "google/siglip-base-patch16-224"))
    app = FastAPI(title="calibrated-cv")

    @app.post("/classify")
    async def classify(file: UploadFile = File(...)) -> dict:
        try:
            image = Image.open(io.BytesIO(await file.read()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="not an image") from exc
        return {**model.predict(encoder.images([image]))[0], "coverage_target": model.coverage}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


def app_factory() -> FastAPI:
    return create_app()
