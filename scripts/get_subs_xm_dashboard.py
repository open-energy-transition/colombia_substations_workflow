#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch dumper for XM/Paratec endpoints.
- Fetch JSON from each endpoint
- Normalize to Feature list
- Recover geometry from properties if missing (lat/lon)
- Write one CSV + one pretty GeoJSON per endpoint
"""

import json, csv, time, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

# ======= EDIT: list of endpoints =======
ENDPOINTS = [
    "https://paratecbackend.xm.com.co/mapas/api/TransmissionMap/getMarkers",
    "https://paratecbackend.xm.com.co/mapas/api/TransmissionMap/getLines",
]
# =======================================

OUTDIR = Path.cwd()
OUTDIR.mkdir(parents=True, exist_ok=True)

TIMEOUT = 30
RETRIES = 3
SLEEP_BETWEEN = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://paratec.xm.com.co/mapa",
    "Origin": "https://paratec.xm.com.co",
}

# ---------------- fetch ----------------
def fetch_json(url: str) -> Any:
    sess = requests.Session()
    for i in range(1, RETRIES + 1):
        try:
            r = sess.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == RETRIES:
                raise
            time.sleep(SLEEP_BETWEEN)

# ------------- feature wrap -------------
def _as_feature(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and obj.get("type") == "Feature":
        return {"type": "Feature",
                "properties": dict(obj.get("properties") or {}),
                "geometry": obj.get("geometry")}
    if isinstance(obj, dict) and ("attributes" in obj or "geometry" in obj):
        props = dict(obj.get("attributes") or {})
        if isinstance(obj.get("properties"), dict):
            props.update(obj["properties"])
        return {"type": "Feature", "properties": props, "geometry": obj.get("geometry")}
    if isinstance(obj, dict):
        return {"type": "Feature", "properties": dict(obj), "geometry": None}
    return {"type": "Feature", "properties": {"value": obj}, "geometry": None}

def to_feature_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        return [_as_feature(f) for f in payload.get("features", [])]
    if isinstance(payload, dict):
        for key in ("features","data","result","results","items","value","lines","Lineas"):
            if isinstance(payload.get(key), list):
                return [_as_feature(x) for x in payload[key]]
        return [_as_feature(payload)]
    if isinstance(payload, list):
        return [_as_feature(x) for x in payload]
    return [_as_feature(payload)]

# ----------- geometry recovery ----------
_LAT_KEYS = ["latitude","lat","LAT","Lat","y","Y"]
_LON_KEYS = ["longitude","lon","lng","long","LON","Lng","x","X"]

def _to_float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def extract_point_from_properties(props: Dict[str, Any]) -> Optional[Tuple[float,float]]:
    lat_key = next((k for k in _LAT_KEYS if k in props), None)
    lon_key = next((k for k in _LON_KEYS if k in props), None)
    if lat_key and lon_key:
        lat, lon = _to_float(props[lat_key]), _to_float(props[lon_key])
        if lat is not None and lon is not None:
            return lon, lat
    return None

def ensure_geometry(feat: Dict[str, Any]) -> Dict[str, Any]:
    geom = feat.get("geometry")
    props = feat.get("properties") or {}
    if isinstance(geom, dict) and geom.get("type") and geom.get("coordinates"):
        props["_geom_from"] = "geometry"
    else:
        pt = extract_point_from_properties(props)
        if pt:
            lon, lat = pt
            feat["geometry"] = {"type":"Point","coordinates":[lon,lat]}
            props["_geom_from"] = "props(lat,lon)"
        else:
            feat["geometry"] = None
            props["_geom_from"] = "none"
    feat["properties"] = props
    return feat

# ---------------- writers ----------------
def write_csv(features: List[Dict[str, Any]], path: Path) -> None:
    rows, header = [], []
    for f in features:
        props = f.get("properties") or {}
        row = dict(props)
        geom = f.get("geometry")
        gtype = geom.get("type") if isinstance(geom, dict) else None
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        row["_geometry_type"] = gtype
        if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["lon"], row["lat"] = coords[0], coords[1]
        row["_geometry_sample"] = json.dumps(coords[:3], ensure_ascii=False) if isinstance(coords, list) else None
        for k in row.keys():
            if k not in header: header.append(k)
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

def write_geojson(features: List[Dict[str, Any]], path: Path) -> None:
    fc = {"type":"FeatureCollection","features":features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- run ----------------
def short_name(url: str) -> str:
    # Use last path part only (after final /), strip query string
    part = url.split("/")[-1].split("?")[0]
    return part or "output"

def export_one(url: str) -> None:
    data = fetch_json(url)
    feats = [ensure_geometry(f) for f in to_feature_list(data)]
    name = short_name(url)
    out_csv, out_geo = OUTDIR / f"{name}.csv", OUTDIR / f"{name}.geojson"
    write_csv(feats, out_csv)
    write_geojson(feats, out_geo)
    print(f"[OK] {url}\n     CSV: {out_csv}\n     GEO: {out_geo}")

def main():
    if not ENDPOINTS:
        sys.exit("Add endpoints first")
    for url in ENDPOINTS:
        try: export_one(url)
        except Exception as e: print(f"[ERROR] {url}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
