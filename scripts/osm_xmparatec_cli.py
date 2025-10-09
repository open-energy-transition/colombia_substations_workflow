
# -*- coding: utf-8 -*-
"""
paratec_osm_union.py

Purpose (OSM → PARATEC pass):
1) Enrich the PARATEC dataset with coordinates: finds substations in OSM by name or location and, if they match a PARATEC substation lacking coordinates, adds the OSM coordinates to PARATEC.
2) Identify OSM substations NOT present in PARATEC (no match by name OR location).
3) Print counts + save CSVs for matched and unmatched.
4) Output the ENRICHED PARATEC dataset (PARATEC schema, coords replaced by OSM when matched).
5) Output a CSV with the MISSING OSM substations from PARATEC (by station name and location).
6) Compute and print % of PARATEC substations covered in OSM.
7) Compute and print count of high-voltage (>57 kV) OSM substations missing from PARATEC.
8) Output a GeoJSON of OSM substations matched to PARATEC with different names for editing.
9) Output a CSV of PARATEC substations with valid coordinates.

This script keeps ALL matching logic inside `matching_utils.py`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import sys, importlib.util
import re
import numpy as np
import pandas as pd
import unicodedata
import csv
import json

# ---------------------- Config ----------------------

# ------------------------------- CLI ----------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="OSM ↔ PARATEC union (paths as args).")
    p.add_argument("--osm", dest="osm_csv", help="Path to OSM dedup CSV (from step 1).")
    p.add_argument("--paratec", dest="paratec_csv", help="Path to PARATEC_enriched_with_XMcoords.csv (from step 6).")
    p.add_argument("--utils-path", dest="utils_path", help="Explicit path to matching_utils.py (optional).")
    p.add_argument("--radius-m", type=float, dest="radius_m", help="Override location match radius (meters).")
    p.add_argument("--fuzzy-threshold", type=int, dest="fuzzy_threshold_override", help="Override fuzzy threshold.")
    p.add_argument("--out-enriched", dest="out_enriched", help="Output CSV for PARATEC_enriched_with_OSMcoords.csv")
    p.add_argument("--out-enriched-location", dest="out_enriched_with_location", help="Output CSV for PARATEC_enriched_with_OSMcoords_with_location.csv")
    p.add_argument("--out-matched", dest="out_matched", help="Output CSV for OSM_to_PARATEC_matched.csv")
    p.add_argument("--out-unmatched-sites", dest="out_unmatched_sites", help="Output CSV for OSM_unmatched_sites.csv")
    p.add_argument("--out-diff-names-geojson", dest="out_diff_names_geojson", help="Output GeoJSON for OSM matched with different names")
    return p.parse_args()

DEFAULTS = {
    "paratec_csv": Path("PARATEC_enriched_with_XMcoords.csv"),
    "osm_csv": Path("osm_substations_dedup.csv"),
    # "outdir": Path("outputs_enriched_paratec_osm"),
    "radius_m": 3000.0,                 # location match radius (meters)
    "use_fuzzy": True,
    "fuzzy_threshold_override": 65,  # None -> use utils.FUZZY_THRESHOLD
    "utils_path": None,                # None -> import from sys.path or same folder
    "high_kv_threshold": 57.0,
}
# ---------------------------------------------------

# ---------- Minimal, robust coordinate parsing (I/O only, not matching logic) ----------
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
def _norm_colname(s: str) -> str:
    # strip accents, lower, remove spaces/underscores
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().replace(" ", "").replace("_", "")

def find_voltage_column(df: pd.DataFrame) -> str | None:
    """Find a likely voltage column in a dataframe."""
    if df is None or df.empty:
        return None
    normalized = {_norm_colname(c): c for c in df.columns}
    candidates = [
        "niveltension", "niveldetension",
        "voltaje", "voltajenominal", "voltage",
        "tension", "tensionnominal",
        "kv", "tensionkv", "niveltensionkv"
    ]
    for key, orig in normalized.items():
        if any(tok in key for tok in candidates):
            return orig
    return None

def _to_float_robust(x) -> float:
    """Parse floats robustly: accept comma decimals, strip degree symbols/labels."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        pass
    s = s.replace("º", "").replace("°", "")
    m = _NUM_RE.search(s)
    if not m:
        return np.nan
    return float(m.group(0).replace(",", "."))

def ensure_lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    """Create/normalize lon/lat columns from common variants."""
    out = df.copy()
    # candidate names (extend if needed)
    lon_cands = ["lon", "longitude", "longitud", "Lon", "LONGITUDE"]
    lat_cands = ["lat", "latitude", "latitud", "Lat", "LATITUDE"]

    if "lon" not in out.columns:
        for c in lon_cands:
            if c in out.columns:
                out["lon"] = out[c].map(_to_float_robust)
                break
    else:
        out["lon"] = out["lon"].map(_to_float_robust)

    if "lat" not in out.columns:
        for c in lat_cands:
            if c in out.columns:
                out["lat"] = out[c].map(_to_float_robust)
                break
    else:
        out["lat"] = out["lat"].map(_to_float_robust)

    return out

def pick_name_col(df: pd.DataFrame, candidates=("elementName","Nombre","name","nombre_subestacion")) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------- Import matching_utils----------
def import_matching_utils(utils_path: Optional[str]):
    if utils_path:
        p = Path(utils_path)
        if not p.exists():
            raise FileNotFoundError(f"--utils-path not found: {utils_path}")
        spec = importlib.util.spec_from_file_location("matching_utils", str(p))
    else:
        try:
            return __import__("matching_utils")
        except Exception:
            here = Path(__file__).resolve().parent
            candidate = here / "matching_utils.py"
            if not candidate.exists():
                raise
            spec = importlib.util.spec_from_file_location("matching_utils", str(candidate))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["matching_utils"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ------------------------------- Main ---------------------------------
def main():
    cfg = DEFAULTS.copy()
    args = parse_args()
    for k in ['osm_csv','paratec_csv','utils_path','radius_m','fuzzy_threshold_override']:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    outdir = Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    mu = import_matching_utils(cfg["utils_path"])
    FUZZY_THRESHOLD = (cfg["fuzzy_threshold_override"]
                       if cfg["fuzzy_threshold_override"] is not None
                       else getattr(mu, "FUZZY_THRESHOLD", 90))

    # Load CSVs
    paratec = pd.read_csv(cfg["paratec_csv"], encoding="utf-8-sig", sep=";")
    osm = pd.read_csv(cfg["osm_csv"], encoding="utf-8")

    # Ensure lon/lat present & numeric
    paratec = ensure_lon_lat(paratec)
    osm = ensure_lon_lat(osm)

    # Basic columns
    paratec_name_col = pick_name_col(paratec, ("Nombre", "elementName", "name"))
    if not paratec_name_col:
        raise ValueError("Could not find PARATEC name column (Nombre/elementName/name).")
    osm_name_col = pick_name_col(osm, ("name_raw", "name", "nombre_subestacion"))
    if not osm_name_col:
        raise ValueError("Could not find OSM name column (name_raw/name/nombre_subestacion).")

    # Use raw Nombre for PARATEC __key (workaround for 499 unique substations)
    paratec["__name_par"] = paratec[paratec_name_col].astype(str).str.strip()
    paratec["__key"] = paratec["__name_par"].astype(str).str.strip()

    # Normalize OSM names (strip voltages, apply key_fn)
    osm["__name_osm"] = osm[osm_name_col].astype(str).map(mu.strip_voltage_tokens).str.strip()
    key_fn = (getattr(mu, "normalized_key_station_plus", None)
              or getattr(mu, "normalized_key_station", None)
              or getattr(mu, "normalized_key_strict"))
    osm["__key"] = osm["__name_osm"].map(lambda s: key_fn(str(s)))

    # Debug: Check unique PARATEC substations
    total_paratec_unique = len(paratec["__key"].dropna().unique())
    print(f"Debug: Total unique PARATEC substations (by __key): {total_paratec_unique}")
    print(f"Debug: Total unique PARATEC names (by Nombre): {len(paratec[paratec_name_col].dropna().unique())}")

    # OSM "sites": one row per (key, lon, lat) to avoid per-voltage duplicates
    osm_sites = (
        osm.dropna(subset=["lon","lat"])
          .drop_duplicates(subset=["__key","lon","lat"], keep="first")
          .reset_index(drop=True)
    )

    # ------------------ OSM → PARATEC matching ------------------
    # We’ll find, for each OSM site, its PARATEC counterpart.
    radius_m = float(cfg["radius_m"])
    osm_idx_to_par: Dict[int, Tuple[str, int, float]] = {}  # osm_idx -> (method, par_idx, aux)

    # 1) LOCATION (Haversine nearest within radius)
    for j, orow in osm_sites.iterrows():
        best_i, best_d = None, float("inf")
        for i, prow in paratec.iterrows():
            if pd.isna(prow["lon"]) or pd.isna(prow["lat"]):
                continue
            d = mu.haversine_m(orow["lat"], orow["lon"], prow["lat"], prow["lon"])
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= radius_m:
            osm_idx_to_par[j] = ("location", best_i, float(best_d))

    # 2) NAME (exact key)
    # Build inverted index from PARATEC keys
    inv_par = {}
    for i, k in paratec["__key"].items():
        if k and k not in inv_par:
            inv_par[k] = i

    for j, row in osm_sites.iterrows():
        if j in osm_idx_to_par:
            continue
        k = row["__key"]
        if k and k in inv_par:
            osm_idx_to_par[j] = ("name_exact", inv_par[k], float("nan"))

    # 3) FUZZY (optional)
    if cfg["use_fuzzy"]:
        for j, row in osm_sites.iterrows():
            if j in osm_idx_to_par:
                continue
            oname = str(row["__name_osm"])
            best_i, best_sc = None, -1.0
            for i, prow in paratec.iterrows():
                pname = str(prow["__name_par"])
                sc = mu.fuzzy_score(oname, pname)
                if sc > best_sc:
                    best_sc, best_i = sc, i
            if best_i is not None and best_sc >= FUZZY_THRESHOLD:
                osm_idx_to_par[j] = ("name_fuzzy", best_i, float(best_sc))

    # ------------------ Build enriched PARATEC (coords from OSM when matched) ------------------
    enriched = paratec.copy()
    # Replace coords on matched PARATEC rows using OSM coords if PARATEC lacks them
    for j, (method, i, aux) in osm_idx_to_par.items():
        # coordinates from this OSM site row
        o_lon = osm_sites.at[j, "lon"]
        o_lat = osm_sites.at[j, "lat"]
        if pd.isna(enriched.at[i, "lon"]) or pd.isna(enriched.at[i, "lat"]):
            enriched.at[i, "lon"] = o_lon
            enriched.at[i, "lat"] = o_lat
        # audit columns (optional, keep in enriched for transparency)
        enriched.at[i, "match_method"] = method
        enriched.at[i, "match_aux"] = aux
        enriched.at[i, "matched_with_osm_name"] = osm_sites.at[j, "__name_osm"]

    # Validate: warn if still NaN coords
    if enriched["lon"].isna().any() or enriched["lat"].isna().any():
        bad = enriched.loc[enriched["lon"].isna() | enriched["lat"].isna(), [paratec_name_col, "lon", "lat"]]
        print(f"Warning: Enriched dataset still contains NaN coords for {len(bad)} rows. Sample:\n{bad.head(10)}")

    # Filter enriched PARATEC for rows with valid coordinates
    enriched_with_location = enriched[enriched["lon"].notna() & enriched["lat"].notna()].copy()

    # ------------------ Matched & Unmatched reporting ------------------
    matched_rows = []
    diff_name_rows = []  # For GeoJSON of matches with different names
    for j, (method, i, aux) in osm_idx_to_par.items():
        osm_name = osm_sites.at[j, "__name_osm"]
        paratec_name = paratec.at[i, "__name_par"]
        matched_rows.append({
            "osm_index": j,
            "osm_name": osm_name,
            "osm_key": osm_sites.at[j, "__key"],
            "paratec_index": i,
            "paratec_name": paratec_name,
            "paratec_key": paratec.at[i, "__key"],
            "method": method,
            "distance_m": aux if method == "location" else np.nan,
            "score": aux if method.startswith("name") else np.nan,
        })
        # Collect matches with different names (location or fuzzy matches)
        if method != "name_exact" and osm_name != paratec_name:
            diff_name_rows.append({
                "osm_index": j,
                "osm_name": osm_name,
                "osm_key": osm_sites.at[j, "__key"],
                "osm_lon": osm_sites.at[j, "lon"],
                "osm_lat": osm_sites.at[j, "lat"],
                "paratec_index": i,
                "paratec_name": paratec_name,
                "paratec_key": paratec.at[i, "__key"],
                "method": method,
                "distance_m": aux if method == "location" else np.nan,
                "score": aux if method.startswith("name") else np.nan,
            })
    df_matched = pd.DataFrame(matched_rows).sort_values(["method","osm_name"]).reset_index(drop=True)

    # Create GeoJSON for matches with different names
    geojson_features = []
    for row in diff_name_rows:
        if pd.isna(row["osm_lon"]) or pd.isna(row["osm_lat"]):
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["osm_lon"]), float(row["osm_lat"])]
            },
            "properties": {
                "osm_index": row["osm_index"],
                "osm_name": row["osm_name"],
                "osm_key": row["osm_key"],
                "paratec_index": row["paratec_index"],
                "paratec_name": row["paratec_name"],
                "paratec_key": row["paratec_key"],
                "match_method": row["method"],
                "distance_m": row["distance_m"] if not pd.isna(row["distance_m"]) else None,
                "score": row["score"] if not pd.isna(row["score"]) else None
            }
        }
        geojson_features.append(feature)
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # ------------------ Unmatched OSM sites with diagnostics ------------------
    unmatched_idx = [j for j in range(len(osm_sites)) if j not in osm_idx_to_par]
    NEARBY_KM = 10.0  # define "nearby" for diagnostics (not used for matching)
    NEARBY_M = NEARBY_KM * 1000.0

    def _nearest_paratec_for(lat, lon):
        best_i, best_d = None, float("inf")
        for i, prow in paratec.iterrows():
            if pd.isna(prow["lon"]) or pd.isna(prow["lat"]):
                continue
            d = mu.haversine_m(lat, lon, prow["lat"], prow["lon"])
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    paratec_key_set = set(paratec["__key"].dropna().tolist())
    OSM_VOLTAGE_COL = find_voltage_column(osm)

    diag_rows = []
    for j in unmatched_idx:
        orow = osm_sites.loc[j]
        oname = str(orow["__name_osm"])
        okey = orow["__key"]
        olon = orow["lon"]
        olat = orow["lat"]

        # Aggregate voltages for this OSM site
        voltages = []
        if OSM_VOLTAGE_COL is not None:
            voltages = sorted(osm.loc[osm["__key"] == okey, OSM_VOLTAGE_COL].dropna().unique().tolist())
        voltages_str = ";".join(str(v) for v in voltages) if voltages else ""

        # Defaults
        nearest_i = None
        nearest_d = np.nan
        nearest_name = None
        nearest_key = None
        key_in_paratec = (okey in paratec_key_set) if isinstance(okey, str) and okey else False

        if pd.isna(olon) or pd.isna(olat):
            reason = "osm_missing_coords"
            best_fuzzy_i = None
            best_fuzzy_name = None
            best_fuzzy_sc = np.nan
        else:
            # Nearest-by-location for context
            nearest_i, nearest_d = _nearest_paratec_for(olat, olon)
            if nearest_i is not None:
                nearest_name = str(paratec.at[nearest_i, "__name_par"])
                nearest_key = paratec.at[nearest_i, "__key"]

            # Best fuzzy candidate (for explanation only)
            best_fuzzy_i, best_fuzzy_sc = None, -1.0
            best_fuzzy_name = None
            if cfg["use_fuzzy"]:
                for i, prow in paratec.iterrows():
                    sc = mu.fuzzy_score(oname, str(prow["__name_par"]))
                    if sc > best_sc:
                        best_sc, best_fuzzy_i = sc, i
                if best_fuzzy_i is not None:
                    best_fuzzy_name = str(paratec.at[best_fuzzy_i, "__name_par"])

            # Decide reason for being unmatched
            if key_in_paratec and (np.isnan(nearest_d) or nearest_d > radius_m):
                reason = "same_name_in_paratec_but_outside_radius"
            elif (best_fuzzy_sc >= FUZZY_THRESHOLD) and (np.isnan(nearest_d) or nearest_d > radius_m):
                reason = "fuzzy_name_match_but_outside_radius"
            elif (not np.isnan(nearest_d)) and nearest_d <= NEARBY_M:
                reason = "nearby_paratec_but_name_mismatch"
            else:
                reason = "no_close_match"

        diag_rows.append({
            "osm_index": j,
            "osm_name": oname,
            "osm_key": okey,
            "osm_lon": olon,
            "osm_lat": olat,
            "voltages_osm": voltages_str,
            "nearest_paratec_index": nearest_i,
            "nearest_paratec_name": nearest_name,
            "nearest_paratec_key": nearest_key,
            "nearest_distance_m": nearest_d,
            "key_in_paratec": key_in_paratec,
            "best_fuzzy_paratec_index": best_fuzzy_i,
            "best_fuzzy_paratec_name": best_fuzzy_name,
            "best_fuzzy_score": best_fuzzy_sc if not np.isnan(best_fuzzy_sc) else np.nan,
            "reason": reason,
        })

    unmatched_osm_sites = pd.DataFrame(diag_rows).sort_values(["reason", "osm_name"]).reset_index(drop=True)

    # ------------------ Stats ------------------
    # % of unique PARATEC substations covered (matched at least once)
    total_paratec_unique = len(paratec["__key"].dropna().unique())
    matched_paratec_idx = set([tpl[1] for tpl in osm_idx_to_par.values()])
    matched_paratec_keys = set(paratec.loc[list(matched_paratec_idx), "__key"].dropna())
    num_matched_par_unique = len(matched_paratec_keys)
    percent_covered = 100.0 * num_matched_par_unique / total_paratec_unique if total_paratec_unique else 0.0

    # Count of high-voltage OSM substations missing from PARATEC
    def max_kv(v):
        try:
            vals = [float(vv.strip()) for vv in str(v).split(';') if vv.strip()]
            return max(vals) / 1000.0 if vals else 0.0
        except Exception:
            return 0.0
    osm["max_kv"] = osm["voltage"].map(max_kv)
    high_osm_keys = set(osm.loc[osm["max_kv"] > cfg["high_kv_threshold"], "__key"].dropna())
    matched_osm_keys = set(osm_sites.loc[list(osm_idx_to_par.keys()), "__key"])
    unmatched_high_osm_keys = high_osm_keys - matched_osm_keys
    num_missing_high = len(unmatched_high_osm_keys)

    # ------------------ Save outputs ------------------
    out_enriched = Path(args.out_enriched) if args.out_enriched else (outdir / "PARATEC_enriched_with_OSMcoords.csv")
    out_enriched_with_location = Path(args.out_enriched_with_location) if args.out_enriched_with_location else (outdir / "PARATEC_enriched_with_OSMcoords_with_location.csv")
    out_matched = Path(args.out_matched) if args.out_matched else (outdir / "OSM_to_PARATEC_matched.csv")
    out_unmatched_sites = Path(args.out_unmatched_sites) if args.out_unmatched_sites else (outdir / "OSM_unmatched_sites.csv")
    out_diff_names_geojson = Path(args.out_diff_names_geojson) if args.out_diff_names_geojson else (outdir / "OSM_matched_different_names.geojson")

    # Drop helper columns from enriched before saving (keep audit if you like)
    save_cols = [c for c in enriched.columns if not c.startswith("__")]
    enriched[save_cols].to_csv(out_enriched, index=False, encoding="utf-8-sig", sep=";", quoting=csv.QUOTE_MINIMAL)
    enriched_with_location[save_cols].to_csv(out_enriched_with_location, index=False, encoding="utf-8-sig", sep=";", quoting=csv.QUOTE_MINIMAL)
    df_matched.to_csv(out_matched, index=False, encoding="utf-8-sig", sep=";")
    unmatched_osm_sites.to_csv(out_unmatched_sites, index=False, encoding="utf-8-sig", sep=";")
    with open(out_diff_names_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    # ------------------ Console report ------------------
    print("\n--- PARATEC OSM Enrichment Report ---")
    print(f"PARATEC rows: {len(paratec)}")
    print(f"PARATEC rows with valid coordinates: {len(enriched_with_location)}")
    print(f"OSM rows: {len(osm)} (sites used: {len(osm_sites)})")
    print(f"Radius (m): {radius_m}")
    print()
    print(f"Matched OSM sites -- PARATEC: {len(df_matched)}")
    print(f"Unmatched OSM sites (record-level, no match by name or location): {len(unmatched_osm_sites)}")
    print(f"OSM sites matched with different names (GeoJSON): {len(geojson_features)}")
    print()
    print(f"Percent of unique PARATEC substations covered in OSM: {percent_covered:.1f}% ({num_matched_par_unique} / {total_paratec_unique})")
    print(f"Number of high-voltage (>{cfg['high_kv_threshold']} kV) OSM substations missing from PARATEC: {num_missing_high}")
    print()
    print("[OK] Saved files:")
    print(f"- {out_enriched}")
    print(f"- {out_enriched_with_location}")
    print(f"- {out_matched}")
    print(f"- {out_unmatched_sites}")
    print(f"- {out_diff_names_geojson}")

if __name__ == "__main__":
    main()
