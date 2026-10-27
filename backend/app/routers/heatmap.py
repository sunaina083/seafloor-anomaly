"""
routers/heatmap.py

GET /tiles/{tile_id}/heatmap -> transparent RGBA PNG (64x64) showing where the
autoencoder failed to reconstruct the tile. The frontend stretches it over the tile.
"""

import io
import importlib.util

import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from . import tiles as tiles_router  # reuse its tile_id -> image path lookup

router = APIRouter(prefix="/tiles", tags=["heatmap"])

PROJECT_ROOT = tiles_router.PROJECT_ROOT
CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "conv_ae.pt"
MODEL_FILE = PROJECT_ROOT / "ml" / "models" / "conv_ae.py"

_model = None
_cache: dict[str, bytes] = {}  # rendered PNGs, at most 700 small files


def get_model():
    """Load the trained autoencoder once, on first request (keeps server startup fast)."""
    global _model
    if _model is None:
        # Load conv_ae.py by file path to avoid clashing with the backend's own `models` module.
        spec = importlib.util.spec_from_file_location("conv_ae_module", MODEL_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.ConvAutoencoder()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def error_to_rgba(err: np.ndarray) -> np.ndarray:
    """Turn a 2D error map into a colored RGBA image: clear where the error is low,
    teal -> amber -> red as it rises."""
    v = err / (err.max() + 1e-8)  # scale to 0..1 within this tile
    stops = [0.0, 0.5, 1.0]
    r = np.interp(v, stops, [45, 251, 255])
    g = np.interp(v, stops, [212, 191, 122])
    b = np.interp(v, stops, [191, 36, 107])
    a = (v ** 1.5) * 0.9 * 255  # the power makes low-error areas fade out
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


@router.get("/{tile_id}/heatmap")
def get_tile_heatmap(tile_id: str):
    if tile_id in _cache:
        return Response(_cache[tile_id], media_type="image/png")

    path = tiles_router._TILE_ID_TO_IMAGE_PATH.get(tile_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Tile image not found")

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    x = torch.from_numpy(img)[None, None]  # shape [1, 1, 64, 64]

    with torch.no_grad():
        recon = get_model()(x)
    err = ((recon - x) ** 2)[0, 0].numpy()  # per-pixel squared error

    buf = io.BytesIO()
    Image.fromarray(error_to_rgba(err), "RGBA").save(buf, format="PNG")
    _cache[tile_id] = buf.getvalue()
    return Response(_cache[tile_id], media_type="image/png")
