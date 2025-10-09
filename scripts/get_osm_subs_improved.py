#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline:
1) RAW: fetch all OSM substations for COUNTRY_ISO.
2) DEDUP BY LOCATION: merge nodes/ways/relations that are spatially close, regardless of name.
   - Keep 1 representative per proximity cluster using a deterministic score.
   - Also carry aggregated fields (name_variants, operator_variants, voltage_variants).
3) FILTER: from the dedup result, keep only rows with a usable name (name_clean != "").
   - Also save a debug CSV of what got filtered out.

Outputs:
- osm_substations_raw.csv
- osm_substations_dedup.csv        
- osm_substations_filtered.csv    (dedup result restricted to name_clean != "")
- osm_substations_dropped_debug.csv

Set COUNTRY_ISO below. No CLI args required.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# =================== Config ===================

COUNTRY_ISO = "CO"  # e.g., "SV" (El Salvador), "CO", "CL"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",  # fallback
]

# Output filenames
OUT_RAW_CSV = "osm_substations_raw.csv"
OUT_FILTERED_CSV = "osm_substations_filtered.csv"
OUT_DEDUP_CSV = "osm_substations_dedup.csv"
OUT_DROPPED_DEBUG_CSV = "osm_substations_dropped_debug.csv"

# Dedup parameters
HAVERSINE_M_THRESHOLD_M = 300.0

# Retry/backoff for Overpass
MAX_TRIES = 4
SLEEP_BASE = 2.0

# =================== Helpers ===================

def norm_space(s: str) -> str:
    s = re.sub(r"[ \t\r\f\v]+", " ", s or "")
    return s.strip()

def clean_name(v: Optional[str]) -> str:
    if v is None:
        return ""
    s = str(v)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)  # control chars
    s = norm_space(s)
    if s.lower() in {"", "nan", "none", "null", "(sin nombre)", "sin nombre"}:
        return ""
    if not re.search(r"[\wÀ-ÿ]", s):
        return ""
    return s

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def score_row_for_keep(r: pd.Series) -> int:
    has_name = 1 if r.get("name_clean") else 0
    has_volt = 1 if r.get("voltage") not in (None, "", float("nan")) else 0
    has_oper = 1 if r.get("operator") not in (None, "", float("nan")) else 0
    has_xy = 1 if (pd.notna(r.get("lat")) and pd.notna(r.get("lon"))) else 0
    tags_count = int(r.get("_tags_count", 0))
    # name and coordinates matter most for a good representative
    return (has_name * 4) + (has_volt * 2) + (has_oper * 1) + (has_xy * 3) + min(tags_count, 10)

# =================== Overpass ===================

def build_query(iso: str) -> str:
    """
    Country-based boundary query (ISO3166-1) for power=substation.
    Includes nodes, ways, and relations; returns center for geometries.
    """
    iso = iso.upper().strip()
    return f"""
[out:json][timeout:1800];
rel["ISO3166-1"="{iso}"]["admin_level"="2"];
map_to_area->.searchArea;
(
  node(area.searchArea)["power"="substation"];
  way(area.searchArea)["power"="substation"];
  relation(area.searchArea)["power"="substation"];
);
out body center tags;
"""

def overpass_request(query: str) -> Dict[str, Any]:
    last_exc = None
    for i in range(1, MAX_TRIES + 1):
        for ep in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(ep, data=query, timeout=180)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(SLEEP_BASE * i)
    raise RuntimeError(f"Overpass failed after {MAX_TRIES} tries: {last_exc}")

# =================== Pipeline ===================

def fase1_download(iso: str) -> pd.DataFrame:
    """Call Overpass and return a normalized DataFrame of elements for the ISO country."""
    query = build_query(iso)
    data = overpass_request(query)

    elements = data.get("elements", [])
    rows = []
    for el in elements:
        t = el.get("type")
        osm_id = el.get("id")
        tags = el.get("tags", {}) or {}

        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            center = el.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")

        name_raw = tags.get("name") or ""
        operator = tags.get("operator") or ""
        voltage = tags.get("voltage") or ""

        rows.append({
            "osm_type": t,
            "osm_id": osm_id,
            "lat": lat,
            "lon": lon,
            "name_raw": name_raw,
            "operator": operator,
            "voltage": voltage,
            "_tags_json": json.dumps(tags, ensure_ascii=False),
            "_tags_count": len(tags),
            "timestamp": el.get("timestamp") or "",
        })

    df = pd.DataFrame(rows)
    df["name_clean"] = df["name_raw"].map(clean_name)

    df.to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote RAW: {OUT_RAW_CSV} ({len(df)} rows)")
    return df

def fase2_dedup_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by PROXIMITY ONLY (ignore names), keeping one representative per cluster.
    Aggregates name/operator/voltage variants for transparency.
    """
    if df.empty:
        df.to_csv(OUT_DEDUP_CSV, index=False, encoding="utf-8-sig")
        print(f"Wrote DEDUP (empty): {OUT_DEDUP_CSV}")
        return df

    g = df.reset_index(drop=True)
    n = len(g)
    visited = [False] * n
    clusters: List[List[int]] = []

    # Simple single-linkage clustering by distance threshold
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        cluster = [i]
        li, lo = g.iloc[i]["lat"], g.iloc[i]["lon"]
        # Walk forward and greedily grab neighbors within threshold
        for j in range(i + 1, n):
            if visited[j]:
                continue
            lj, ljo = g.iloc[j]["lat"], g.iloc[j]["lon"]
            if pd.notna(li) and pd.notna(lo) and pd.notna(lj) and pd.notna(ljo):
                d = haversine_m(li, lo, lj, ljo)
                if d <= HAVERSINE_M_THRESHOLD_M:
                    visited[j] = True
                    cluster.append(j)
        clusters.append(cluster)

    keep_rows = []
    for idxs in clusters:
        sub = g.iloc[idxs].copy()

        # pick representative
        sub["_score"] = sub.apply(score_row_for_keep, axis=1)

        if "timestamp" in sub.columns:
            ts = pd.to_datetime(sub["timestamp"], errors="coerce")
            sub["_ts"] = ts.fillna(pd.Timestamp(0))
        else:
            sub["_ts"] = pd.Timestamp(0)

        if "osm_id" in sub.columns:
            try:
                sub["_osm_id_num"] = pd.to_numeric(sub["osm_id"], errors="coerce").fillna(0).astype(int)
            except Exception:
                sub["_osm_id_num"] = 0
        else:
            sub["_osm_id_num"] = 0

        sub = sub.sort_values(by=["_score", "_ts", "_osm_id_num"], ascending=[False, False, True])
        rep = sub.iloc[0].copy()

        # aggregate variants for transparency
        def uniq_nonempty(series):
            vals = [str(x).strip() for x in series if isinstance(x, str) and str(x).strip() != ""]
            out = sorted(set(vals))
            return "; ".join(out) if out else ""

        rep["name_variants"] = uniq_nonempty(sub["name_raw"])
        rep["operator_variants"] = uniq_nonempty(sub["operator"])
        rep["voltage_variants"] = uniq_nonempty(sub["voltage"])

        keep_rows.append(rep)

    dedup = pd.DataFrame(keep_rows).reset_index(drop=True)
    dedup.to_csv(OUT_DEDUP_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote DEDUP (location-based): {OUT_DEDUP_CSV} ({len(dedup)} rows)")
    return dedup

def fase3_filter_named(df_dedup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the DEDUP result, keep only rows with a usable name_clean.
    Also write a debug CSV with dropped rows.
    """
    if "name_clean" not in df_dedup.columns:
        df_dedup["name_clean"] = df_dedup["name_raw"].map(clean_name)

    bad_mask = (df_dedup["name_clean"].isna()) | (df_dedup["name_clean"] == "")
    debug_bad = df_dedup.loc[bad_mask].copy()
    filtered = df_dedup.loc[~bad_mask].copy()

    debug_bad.to_csv(OUT_DROPPED_DEBUG_CSV, index=False, encoding="utf-8-sig")
    print(f"Dropped after dedup due to empty name: {len(debug_bad)} -> {OUT_DROPPED_DEBUG_CSV}")

    filtered.to_csv(OUT_FILTERED_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote FILTERED (post-dedup): {OUT_FILTERED_CSV} ({len(filtered)} rows)")

    return filtered, debug_bad

# =================== Main ===================

def main():
    raw = fase1_download(COUNTRY_ISO)
    dedup = fase2_dedup_location(raw)
    _filtered, _debug = fase3_filter_named(dedup)

if __name__ == "__main__":
    main()
