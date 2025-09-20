#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ALL substations shown on PARATEC's map as GeoJSON (EPSG:4326) for JOSM.

Outputs:
  - subestaciones_upme.geojson (pretty-printed)
  - subestaciones_upme.csv
"""

import json
import math
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv

BASE = "https://geo.upme.gov.co/server/rest/services/Capas_EnergiaElectrica/sistema_transmision_subestaciones_construidas/FeatureServer/18"

def session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=0.4, status_forcelist=(429,500,502,503,504), allowed_methods={"GET"})
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "paratec-substations-export/1.0"})
    return s

def get_ids(s: requests.Session):
    r = s.get(f"{BASE}/query", params={
        "where": "1=1",
        "returnIdsOnly": "true",
        "f": "json",
    }, timeout=90)
    r.raise_for_status()
    data = r.json()
    ids = data.get("objectIds") or []
    ids.sort()
    return ids

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_chunk_geojson(s: requests.Session, ids):
    r = s.get(f"{BASE}/query", params={
        "objectIds": ",".join(map(str, ids)),
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,          # lon/lat
        "f": "geojson",         # GeoJSON directly
    }, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    s = session()
    print("Getting OBJECTIDs")
    ids = get_ids(s)
    if not ids:
        raise SystemExit("No object IDs returned; layer empty or access issue.")

    CHUNK = 1800
    all_feats = []

    print(f"Downloading in {math.ceil(len(ids)/CHUNK)} chunk(s)")
    for i, chunk in enumerate(chunked(ids, CHUNK), 1):
        data = fetch_chunk_geojson(s, chunk)
        feats = data.get("features") or []
        all_feats.extend(feats)
        print(f"  chunk {i}: {len(feats)} features")
        time.sleep(0.1)

    fc = {
        "type": "FeatureCollection",
        "features": all_feats,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }

    # Pretty-print GeoJSON for readability
    with open("subestaciones_upme.geojson", "w", encoding="utf-8", newline="\n") as f:
        json.dump(fc, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print(f"Saved subestaciones_upme.geojson ({len(all_feats)} features)")

    # Write a CSV with lon/lat + key attributes
    rows = []
    for ftr in all_feats:
        props = ftr.get("properties") or {}
        geom = ftr.get("geometry") or {}
        coords = (geom.get("coordinates") or [None, None])
        rows.append({
            "nombre_subestacion": props.get("nombre_subestacion"),
            "sistema": props.get("sistema"),
            "tension_kV": props.get("tension"),
            "departamento": props.get("departamento"),
            "municipio": props.get("municipio"),
            "lon": coords[0],
            "lat": coords[1],
            # keep raw attributes:
            "id_organizacion": props.get("id_organizacion"),
            "nombre_propietario": props.get("nombre_propietario"),
            "fecha_operacion": props.get("fecha_operacion"),
            "observacion": props.get("observacion"),
        })

    df = pd.DataFrame(rows)

    # Drop rows where nombre_subestacion is missing/empty
    df = df[df["nombre_subestacion"].notna() & (df["nombre_subestacion"].str.strip() != "")]

    # Sanitize text columns: remove internal newlines to avoid split rows
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(
        lambda col: col.str.replace(r"[\r\n]+", " ", regex=True)
    ).apply(lambda col: col.str.strip())

    # Save CSV with all fields quoted (safe for commas, quotes, etc.)
    df.to_csv(
        "subestaciones_upme.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )
    print(f"Saved subestaciones_upme.csv ({len(df)} rows)")

if __name__ == "__main__":
    main()
