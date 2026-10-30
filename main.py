"""
Cuddly Viewer - backend (Fabric & Spatial Realism Edition)
"""

import colorsys
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cuddly-viewer")

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = BASE_DIR / "index.html"

QLOO_API_KEY = os.getenv("QLOO_API_KEY", "").strip()
QLOO_BASE_URL = os.getenv("QLOO_BASE_URL", "https://hackathon.api.qloo.com").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Fixed to official, high-speed active models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash").strip()

QLOO_TIMEOUT_SECONDS = 8

# Proportional scaling limits for standard 3D meshes (1.0 = original mesh size)
DEFAULT_SHAPE = {
    "width": 1.0,
    "height": 1.0,
    "depth": 1.0,
    "color": "#3b82f6",
    "fabric_type": "cotton",
    "fabric_name": "Heavyweight French Terry Cotton",
    "sheen": 0.1,
    "roughness": 0.8
}
LIMITS = {"width": (0.8, 1.35), "height": (0.85, 1.25), "depth": (0.85, 1.25)}
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

CACHE_MAX_ITEMS = 100
_cache: dict = {}


class QlooError(Exception):
    pass


def qloo_get(path: str, params: dict) -> Any:
    if not QLOO_API_KEY:
        raise QlooError("QLOO_API_KEY is missing from .env.")

    try:
        response = requests.get(
            QLOO_BASE_URL + path,
            headers={"X-Api-Key": QLOO_API_KEY, "Accept": "application/json"},
            params=params,
            timeout=QLOO_TIMEOUT_SECONDS,
        )
    except requests.RequestException as err:
        raise QlooError(f"Could not reach Qloo ({type(err).__name__}).") from err

    if response.status_code in (401, 403):
        raise QlooError("Qloo rejected the API key.")
    if response.status_code == 404:
        return {}
    if not response.ok:
        raise QlooError(f"Qloo returned HTTP {response.status_code}.")

    try:
        return response.json()
    except ValueError as err:
        raise QlooError("Qloo sent invalid JSON.") from err


def entity_list(payload: Any) -> list:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        results = payload.get("results", payload)
        if isinstance(results, dict):
            results = results.get("entities", [])
        items = results if isinstance(results, list) else []
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def tag_names(entity: dict, limit: int = 8) -> list:
    names = []
    for tag in entity.get("tags") or []:
        if isinstance(tag, dict) and tag.get("name"):
            names.append(str(tag["name"]))
        elif isinstance(tag, str):
            names.append(tag)
    return names[:limit]


def fetch_taste_dna(brand: str) -> dict:
    dna = {"used": False, "matched_name": None, "tags": [], "related_brands": [], "warnings": []}
    found = []
    last_error = None

    for params in ({"query": brand, "types": "urn:entity:brand"}, {"query": brand}):
        try:
            found = entity_list(qloo_get("/search", params))
        except QlooError as err:
            last_error = err
            continue
        if found:
            break

    if not found:
        dna["warnings"].append(str(last_error) if last_error else f"Qloo has no match for “{brand}”.")
        return dna

    top = found[0]
    dna["used"] = True
    dna["matched_name"] = top.get("name") or brand
    dna["tags"] = tag_names(top)

    entity_id = top.get("entity_id") or top.get("id")
    if entity_id:
        try:
            related = entity_list(
                qloo_get(
                    "/v2/insights",
                    {"filter.type": "urn:entity:brand", "signal.interests.entities": entity_id, "take": 6},
                )
            )
            dna["related_brands"] = [str(item["name"]) for item in related if item.get("name")][:6]
        except QlooError as err:
            dna["warnings"].append(f"Related brands unavailable: {err}")

    return dna


# ---------------------------------------------------------------------------
# Gemini PBR & Fabric Intelligence
# ---------------------------------------------------------------------------
class HoodieShape(BaseModel):
    width_scale: float = Field(description="Hoodie width: 0.85 (slim) to 1.30 (oversized/boxy)")
    height_scale: float = Field(description="Hoodie height: 0.85 (cropped) to 1.25 (long line)")
    depth_scale: float = Field(description="Hoodie depth: 0.85 to 1.25")
    color_hex: str = Field(description="Hex color reflecting the brand mood (#RRGGBB)")
    fabric_type: str = Field(description="One of: 'silk', 'cotton', 'fleece', 'satin', 'leather', 'nylon'")
    fabric_name: str = Field(description="Full fabric description, e.g. '100% Charmeuse Silk', 'Heavyweight French Terry Cotton'")
    sheen: float = Field(description="0.0 for matte/fleece up to 1.0 for lustrous silk/satin")
    roughness: float = Field(description="0.2 for smooth silk to 0.85 for textured fleece/cotton")
    reasoning: str = Field(description="Max 25 words explaining the fit, silhouette and fabric choice")


_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def shape_with_gemini(brand: str, dna: dict) -> dict:
    prompt = (
        "You are an expert luxury & streetwear fashion director configuring a 3D showroom hoodie.\n"
        f"Brand searched: {brand}\n"
        f"Qloo matched name: {dna['matched_name'] or 'none'}\n"
        f"Qloo taste tags: {', '.join(dna['tags']) or 'none'}\n"
        f"Qloo related brands: {', '.join(dna['related_brands']) or 'none'}\n\n"
        "Analyze the brand's aesthetic. If the brand is high luxury or avant-garde (e.g. Dior, Gucci, Casablanca), "
        "opt for silk or satin with high sheen (0.8 - 1.0) and lower roughness (0.25 - 0.35). "
        "If it is rugged streetwear or skate (e.g. Nike, Stussy, Supreme), choose heavyweight fleece/cotton with low sheen (0.05 - 0.2) and high roughness (0.75 - 0.9)."
    )
    last_error = None
    for model in dict.fromkeys([GEMINI_MODEL, GEMINI_FALLBACK_MODEL]):
        if not model:
            continue
        try:
            response = _gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=HoodieShape,
                ),
            )
            if not response.text:
                raise ValueError("Gemini returned empty text.")
            shape = HoodieShape.model_validate_json(response.text)
            break
        except Exception as err:
            log.warning("Gemini model %s failed: %s", model, err)
            last_error = err
    else:
        raise last_error or ValueError("No Gemini model configured.")

    color = shape.color_hex.strip()
    return {
        "width": clamp(shape.width_scale, *LIMITS["width"]),
        "height": clamp(shape.height_scale, *LIMITS["height"]),
        "depth": clamp(shape.depth_scale, *LIMITS["depth"]),
        "color": color if HEX_COLOR.match(color) else DEFAULT_SHAPE["color"],
        "fabric_type": shape.fabric_type.lower(),
        "fabric_name": shape.fabric_name.strip()[:60],
        "sheen": clamp(shape.sheen, 0.0, 1.0),
        "roughness": clamp(shape.roughness, 0.15, 0.95),
        "reasoning": shape.reasoning.strip()[:200],
    }


def shape_with_rules(brand: str, dna: dict) -> dict:
    text = " ".join([brand] + dna["tags"]).lower()
    width, height, depth = 1.0, 1.0, 1.0
    fabric_type = "cotton"
    fabric_name = "Premium Combed Cotton"
    sheen = 0.1
    roughness = 0.8
    reasoning = "Standard streetwear cut and cotton weave."

    if any(w in text for w in ("silk", "luxury", "satin", "formal", "dior", "gucci", "chanel")):
        width, depth = 0.95, 0.95
        fabric_type = "silk"
        fabric_name = "100% Mulberry Silk Charmeuse"
        sheen = 0.95
        roughness = 0.25
        reasoning = "Luxury signals indicate a fluid silhouette crafted in lustrous silk."
    elif any(w in text for w in ("streetwear", "skate", "boxy", "oversized", "supreme", "nike")):
        width, depth = 1.15, 1.1
        fabric_type = "fleece"
        fabric_name = "450 GSM Heavyweight Fleece"
        sheen = 0.05
        roughness = 0.85
        reasoning = "Streetwear heritage points toward an oversized fit in heavyweight fleece."

    hue = int(hashlib.md5(brand.lower().encode("utf-8")).hexdigest()[:4], 16) % 360 / 360
    red, green, blue = colorsys.hls_to_rgb(hue, 0.55, 0.7)
    color = "#{:02x}{:02x}{:02x}".format(int(red * 255), int(green * 255), int(blue * 255))

    return {
        "width": width,
        "height": height,
        "depth": depth,
        "color": color,
        "fabric_type": fabric_type,
        "fabric_name": fabric_name,
        "sheen": sheen,
        "roughness": roughness,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cuddly Viewer - Spatial Showroom")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
@app.get("/showroom", include_in_schema=False)
def serve_index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="index.html not found.")
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "qloo_key_set": bool(QLOO_API_KEY),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "model_exists": (STATIC_DIR / "hoodie.glb").exists(),
    }


@app.get("/get-product-style")
def get_product_style(brand_name: str = Query(..., min_length=1, max_length=60)):
    brand = " ".join(brand_name.split())
    if not brand:
        raise HTTPException(status_code=400, detail="brand_name cannot be empty.")

    cache_key = brand.lower()
    if cache_key in _cache:
        return _cache[cache_key]

    dna = fetch_taste_dna(brand)
    warnings = list(dna["warnings"])
    shape = None
    shape_source = "rules"

    if _gemini_client:
        try:
            shape = shape_with_gemini(brand, dna)
            shape_source = "gemini"
        except Exception as err:
            log.warning("Gemini failed: %s", err)
            warnings.append("Gemini temporarily unavailable; used smart fallback rules.")
    else:
        warnings.append("GEMINI_API_KEY missing; used smart fallback rules.")

    if shape is None:
        shape = shape_with_rules(brand, dna)

    result = {
        "target_category": "Streetwear Hoodie",
        "spatial_showroom_commands": {
            "scale_width": shape["width"],
            "scale_height": shape["height"],
            "scale_depth": shape["depth"],
            "target_color": shape["color"],
            "fabric_type": shape["fabric_type"],
            "fabric_name": shape["fabric_name"],
            "sheen": shape["sheen"],
            "roughness": shape["roughness"],
        },
        "meta": {
            "brand_query": brand,
            "qloo_used": dna["used"],
            "qloo_matched_name": dna["matched_name"],
            "qloo_tags": dna["tags"],
            "qloo_related_brands": dna["related_brands"],
            "shape_source": shape_source,
            "reasoning": shape["reasoning"],
            "warnings": warnings,
        },
    }

    if dna["used"] and shape_source == "gemini":
        if len(_cache) >= CACHE_MAX_ITEMS:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = result

    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
