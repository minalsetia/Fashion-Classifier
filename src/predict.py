"""Inference: classify one image or a batch, logging every run to the DB.

A single Predictor loads both trained models once and reuses them. Each call
returns a plain dict of predictions and (by default) writes a tracking row to
the SQLite database so the UI's history view always reflects what happened.

Programmatic use:
    from src.predict import Predictor
    p = Predictor()
    print(p.predict_single("data/images/123.jpg"))
    print(p.predict_batch(["https://.../a.jpg", "data/images/123.jpg"]))

CLI:
    python -m src.predict path/or/url [more ...]
"""
from __future__ import annotations

import argparse
import io
import sys

import requests
import torch
from PIL import Image

from . import config, database, model as model_lib

# Combined version string recorded with every row.
COMBINED_VERSION = f"{config.MODEL_VERSION['gender']}+{config.MODEL_VERSION['sleeve']}"


def load_image(source: str) -> Image.Image:
    """Open an image from a local path or an http(s) URL as RGB."""
    if str(source).lower().startswith(("http://", "https://")):
        resp = requests.get(source, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


class Predictor:
    """Loads both task models once and predicts gender + sleeve for images."""

    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.transform = model_lib.get_transforms(train=False)
        self.models: dict[str, tuple] = {}  # task -> (net, classes)
        self._load("gender", config.GENDER_MODEL_PATH)
        self._load("sleeve", config.SLEEVE_MODEL_PATH)

    def _load(self, task: str, path) -> None:
        if path.exists():
            self.models[task] = model_lib.load_model(path, self.device)
        else:
            print(f"WARNING: {task} model not found at {path}. "
                  f"Train it with `python -m src.train --task {task}`.")

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def _classify(self, task: str, tensor: torch.Tensor) -> tuple[str | None, float | None]:
        if task not in self.models:
            return None, None
        net, classes = self.models[task]
        with torch.no_grad():
            probs = torch.softmax(net(tensor), dim=1)[0]
        idx = int(probs.argmax())
        return classes[idx], round(float(probs[idx]), 4)

    def predict_image(self, img: Image.Image) -> dict:
        """Predict both attributes for an already-loaded PIL image."""
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        gender, gender_conf = self._classify("gender", tensor)
        sleeve, sleeve_conf = self._classify("sleeve", tensor)
        return {
            "predicted_gender": gender,
            "gender_confidence": gender_conf,
            "predicted_sleeve": sleeve,
            "sleeve_confidence": sleeve_conf,
            "model_version": COMBINED_VERSION,
        }

    def _predict_and_log(self, source: str, run_id: str, run_type: str,
                         log: bool) -> dict:
        """Predict one source, returning a result dict; log to DB if requested."""
        result = {"image_ref": source, "run_id": run_id, "run_type": run_type}
        try:
            preds = self.predict_image(load_image(source))
            result.update(preds)
            result["status"] = "success"
            result["error_message"] = None
        except Exception as exc:  # bad URL, decode error, etc.
            result.update({
                "predicted_gender": None, "gender_confidence": None,
                "predicted_sleeve": None, "sleeve_confidence": None,
                "model_version": COMBINED_VERSION,
                "status": "error", "error_message": str(exc),
            })
        if log:
            database.log_prediction(
                run_id=run_id, run_type=run_type, image_ref=source,
                predicted_gender=result["predicted_gender"],
                gender_confidence=result["gender_confidence"],
                predicted_sleeve=result["predicted_sleeve"],
                sleeve_confidence=result["sleeve_confidence"],
                model_version=result["model_version"],
                status=result["status"], error_message=result["error_message"],
            )
        return result

    def predict_single(self, source: str, log: bool = True) -> dict:
        run_id = database.new_run_id()
        return self._predict_and_log(source, run_id, "single", log)

    def predict_batch(self, sources: list[str], log: bool = True) -> list[dict]:
        run_id = database.new_run_id()  # one id groups the whole batch
        return [self._predict_and_log(s, run_id, "batch", log) for s in sources]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify image(s) by path or URL.")
    parser.add_argument("sources", nargs="+", help="image path(s) or URL(s)")
    args = parser.parse_args()

    predictor = Predictor()
    if not predictor.ready:
        print("No trained models available. Train first, then retry.")
        sys.exit(1)

    if len(args.sources) == 1:
        results = [predictor.predict_single(args.sources[0])]
    else:
        results = predictor.predict_batch(args.sources)

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
