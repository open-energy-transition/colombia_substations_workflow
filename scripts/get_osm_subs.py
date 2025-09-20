#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
import math
import pandas as pd
import json
from collections import defaultdict

# =================== Config ===================

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",  # fallback
]

QUERY = r"""
[out:json][timeout:1800];

rel["ISO3166-1"="CO"]["admin_level"="2"];
map_to_area->.searchArea;

(
  node(area.searchArea)["power"="substation"];
  way(area.searchArea)["power"="substation"];
  relation(area.searchArea)["power"="substation"];
);

out center;
"""

# =================== Phase 1: RAW download ===================

def fetch_overpass(query: str) -> dict:
    last_exc = None
    for i, url in enumerate(OVERPASS_ENDPOINTS):
        try:
            resp = requests.post(url, data={"data": query}, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            time.sleep(1.0 + i * 1.5)
    raise last_exc

def extract_coords(el):
    if "lat" in el and "lon" in el:
        return float(el["lon"]), float(el["lat"])
    c = el.get("center", {})
    if isinstance(c, dict) and ("lat" in c and "lon" in c):
        return float(c["lon"]), float(c["lat"])
    return None, None

def fase1_download():
    print("=== PHASE 1: Download OSM substations (Colombia) ===")
    data = fetch_overpass(QUERY)
    elements = data.get("elements", [])
    print(f"Raw elements downloaded: {len(elements)}")

    features = []
    skipped = 0
    for el in elements:
        lon, lat = extract_coords(el)
        if lon is None or lat is None:
            skipped += 1
            continue
        t = el.get("type")
        osm_id = el.get("id")
        tags = el.get("tags", {}) or {}
        props = {
            "name": tags.get("name", ""),
            "operator": tags.get("operator", ""),
            "substation": tags.get("substation", ""),
            "voltage": tags.get("voltage", ""),
        }
        features.append({
            "osm_type": t,
            "osm_id": osm_id,
            "lon": lon,
            "lat": lat,
            "props": props,
        })

    print(f"Features with valid coordinates: {len(features)}")
    if skipped > 0:
        print(f"Note: {skipped} elements skipped due to missing coordinates")
    return features

# =================== Phase 2: Deduplication ===================

def norm(s):
    if not s:
        return None
    return " ".join(s.strip().lower().split())

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def merge_props(props_list):
    pick_first = ["name", "operator", "substation", "voltage"]
    merged = {k: "" for k in pick_first}
    for p in props_list:
        for k in pick_first:
            v = (p.get(k) or "").strip()
            if v and not merged[k]:
                merged[k] = v
    return merged

def spatial_cluster(feats, radius_m=100.0):
    unused = set(range(len(feats)))
    clusters = []
    while unused:
        i = unused.pop()
        base = feats[i]
        group = [base]
        to_remove = []
        for j in list(unused):
            f = feats[j]
            d = haversine_m(base["lat"], base["lon"], f["lat"], f["lon"])
            if d <= radius_m:
                group.append(f)
                to_remove.append(j)
        for j in to_remove:
            unused.remove(j)

        rep = None
        for typ in ["way", "relation", "node"]:
            for g in group:
                if g["osm_type"] == typ:
                    rep = g
                    break
            if rep:
                break
        if rep is None:
            rep = group[0]

        lon, lat = rep["lon"], rep["lat"]
        props = merge_props([g["props"] for g in group])
        osm_ids = [f'{g["osm_type"]}/{g["osm_id"]}' for g in group]

        clusters.append({
            "lon": lon,
            "lat": lat,
            "count": len(group),
            "osm_ids": "; ".join(osm_ids),
            "osm_types": "; ".join(sorted(set(g["osm_type"] for g in group))),
            **props
        })
    return clusters

def dedup_substations(features):
    named = defaultdict(list)
    unnamed = []
    for f in features:
        nm = norm(f["props"].get("name"))
        if nm:
            named[nm].append(f)
        else:
            unnamed.append(f)

    clusters = []
    for feats in named.values():
        clusters.extend(spatial_cluster(feats, radius_m=500.0))
    if unnamed:
        clusters.extend(spatial_cluster(unnamed, radius_m=50.0))
    return clusters

def df_to_geojson(df_subset, out_path):
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": row.drop(["lon","lat"]).to_dict(),
            "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
        } for _, row in df_subset.iterrows()]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

def clean_text(series: pd.Series) -> pd.Series:
    """Normalize for storage."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"[\u200B-\u200D\u2060\ufeff]", "", regex=True)  # zero-width/BOM
        .str.replace("\u00A0", " ", regex=False)  # NBSP
        .str.strip()
    )

def is_effectively_empty(series: pd.Series) -> pd.Series:
    """Detect emptiness after removing whitespace/hidden chars."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"[\u200B-\u200D\u2060\ufeff]", "", regex=True)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .eq("")
    )

def fase2_dedup(features):
    print("\n=== PHASE 2: Deduplicate substations ===")
    print(f"Input features: {len(features)}")
    clusters = dedup_substations(features)
    print(f"Deduplicated clusters: {len(clusters)}")

    df = pd.DataFrame(clusters)

    # Drop *_all columns if any
    drop_cols = [c for c in df.columns if c.endswith("_all")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Save full dedup CSV
    full_csv = "osm_colombia_substations_dedup.csv"
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"Deduplicated CSV (FULL) saved: {full_csv} ({len(df)} rows)")

    # Clean name and voltage
    df["name"] = clean_text(df["name"])
    df["voltage"] = clean_text(df["voltage"])

    has_name = ~is_effectively_empty(df["name"])
    has_voltage = ~is_effectively_empty(df["voltage"])

    # GeoJSONs for fixing in JOSM
    named_no_voltage = df[has_name & (~has_voltage)]
    voltage_no_name = df[has_voltage & (~has_name)]
    df_to_geojson(named_no_voltage, "osm_substations_named_no_voltage.geojson")
    print(f"GeoJSON saved (name without voltage): {len(named_no_voltage)} features")
    df_to_geojson(voltage_no_name, "osm_substations_voltage_no_name.geojson")
    print(f"GeoJSON saved (voltage without name): {len(voltage_no_name)} features")

    # Final filter: keep only BOTH name and voltage
    df_filtered = df[has_name & has_voltage].copy()
    filtered_csv = "osm_substations_filtered.csv"
    df_filtered.to_csv(filtered_csv, index=False, encoding="utf-8-sig")
    print(f"Final CSV (only BOTH name AND voltage) saved: {filtered_csv} ({len(df_filtered)} rows)")

    # Debug dump of dropped rows
    dropped_mask = ~(has_name & has_voltage)
    debug_bad = df[dropped_mask]
    if not debug_bad.empty:
        debug_csv = "osm_substations_dropped_debug.csv"
        debug_bad.to_csv(debug_csv, index=False, encoding="utf-8-sig")
        print(f"Dropped rows written to {debug_csv}: {len(debug_bad)}")

    return clusters

# =================== Main ===================

def main():
    features = fase1_download()
    _ = fase2_dedup(features)

if __name__ == "__main__":
    main()
