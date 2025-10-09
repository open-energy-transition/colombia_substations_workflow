# -*- coding: utf-8 -*-
"""
xm_upme_enrich_from_upme.py

Purpose (UPME → XM pass):
1) Inspect UPME substations and match them with XM:
   - first by LOCATION (Haversine, configurable radius),
   - then by NAME (exact key using utils),
   - then FUZZY (utils.fuzzy_score).
   If matched, REPLACE the XM location with the UPME location.

2) Identify UPME substations NOT present in XM (name-level, voltage stripped).

3) Print counts + save CSVs for matched and unmatched.

4) Output the ENRICHED XM dataset (XM schema, but coords replaced by UPME when matched).

5) Output a CSV with the MISSING UPME substations from XM (by station name).

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

# ---------------------- Config ----------------------
DEFAULTS = {
    "xm_csv": Path("getMarkers.csv"),
    "upme_csv": Path("subestaciones_upme.csv"),
    # "outdir": Path("outputs_enriched"),
    "radius_m": 3000.0,                 # location match radius (meters)
    "use_fuzzy": True,
    "fuzzy_threshold_override": 65,  # None -> use utils.FUZZY_THRESHOLD
    "utils_path": None,                # None -> import from sys.path or same folder
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

# ------------------------------- CLI ----------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="XM ⟵ UPME enrichment (paths as args).")
    p.add_argument("--xm", dest="xm_csv", help="Path to XM markers CSV (getMarkers.csv).")
    p.add_argument("--upme", dest="upme_csv", help="Path to UPME CSV (subestaciones_upme.csv).")
    p.add_argument("--utils-path", dest="utils_path", help="Explicit path to matching_utils.py (optional).")
    p.add_argument("--radius-m", type=float, dest="radius_m", help="Location match radius in meters.")
    p.add_argument("--fuzzy-threshold", type=int, dest="fuzzy_threshold_override", help="Override fuzzy threshold (default from utils).")
    p.add_argument("--out-enriched", dest="out_enriched", help='Output CSV for enriched XM (default "XM_enriched_with_UPME_coords.csv").')
    p.add_argument("--out-matched", dest="out_matched", help='Output CSV for matched pairs (default "UPME_to_XM_matched.csv").')
    p.add_argument("--out-unmatched", dest="out_unmatched_sites", help='Output CSV for unmatched UPME sites (default "UPME_unmatched_sites.csv").')
    return p.parse_args()

# ------------------------------- Main ---------------------------------
def main():
    cfg = DEFAULTS.copy()
    args = parse_args()
    # Override defaults with provided CLI args (only if given)
    for k in ['xm_csv','upme_csv','utils_path','radius_m','fuzzy_threshold_override']:
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
    xm = pd.read_csv(cfg["xm_csv"], encoding="utf-8")
    up = pd.read_csv(cfg["upme_csv"], encoding="utf-8")

    # Ensure lon/lat present & numeric
    xm = ensure_lon_lat(xm)
    up = ensure_lon_lat(up)

    # Basic columns
    xm_name_col = pick_name_col(xm, ("elementName","Nombre","name","nombre_subestacion"))
    if not xm_name_col:
        raise ValueError("Could not find XM name column (elementName/Nombre/name/nombre_subestacion).")
    if "nombre_subestacion" not in up.columns:
        raise ValueError("UPME CSV must contain 'nombre_subestacion'.")

    # Voltage-stripped display names
    xm["__name_xm"] = xm[xm_name_col].astype(str).map(mu.strip_voltage_tokens).str.strip()
    up["__name_upme"] = up["nombre_subestacion"].astype(str).map(mu.strip_voltage_tokens).str.strip()

    # Canonical station keys (heavy normalization)
    key_fn = (getattr(mu, "normalized_key_station_plus", None)
              or getattr(mu, "normalized_key_station", None)
              or getattr(mu, "normalized_key_strict"))

    xm["__key"] = xm["__name_xm"].map(lambda s: key_fn(str(s)))
    up["__key"] = up["__name_upme"].map(lambda s: key_fn(str(s)))

    # UPME "sites": one row per (key, lon, lat) to avoid per-voltage duplicates
    up_sites = (
        up.dropna(subset=["lon","lat"])
          .drop_duplicates(subset=["__key","lon","lat"], keep="first")
          .reset_index(drop=True)
    )

    # ------------------ UPME → XM matching ------------------
    # We’ll find, for each UPME site, its XM counterpart.
    radius_m = float(cfg["radius_m"])
    upme_idx_to_xm: Dict[int, Tuple[str, int, float]] = {}  # up_idx -> (method, xm_idx, aux)

    # 1) LOCATION (Haversine nearest within radius)
    for j, ur in up_sites.iterrows():
        best_i, best_d = None, float("inf")
        for i, xr in xm.iterrows():
            if pd.isna(xr["lon"]) or pd.isna(xr["lat"]):
                continue
            d = mu.haversine_m(ur["lat"], ur["lon"], xr["lat"], xr["lon"])
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= radius_m:
            upme_idx_to_xm[j] = ("location", best_i, float(best_d))

    # 2) NAME (exact key)
    # Build inverted index from XM keys
    inv_xm = {}
    for i, k in xm["__key"].items():
        if k and k not in inv_xm:
            inv_xm[k] = i

    for j, row in up_sites.iterrows():
        if j in upme_idx_to_xm:
            continue
        k = row["__key"]
        if k and k in inv_xm:
            upme_idx_to_xm[j] = ("name_exact", inv_xm[k], float("nan"))

    # 3) FUZZY (optional)
    if cfg["use_fuzzy"]:
        for j, row in up_sites.iterrows():
            if j in upme_idx_to_xm:
                continue
            uname = str(row["__name_upme"])
            best_i, best_sc = None, -1.0
            for i, xr in xm.iterrows():
                xname = str(xr["__name_xm"])
                sc = mu.fuzzy_score(uname, xname)
                if sc > best_sc:
                    best_sc, best_i = sc, i
            if best_i is not None and best_sc >= FUZZY_THRESHOLD:
                upme_idx_to_xm[j] = ("name_fuzzy", best_i, float(best_sc))

    # ------------------ Build enriched XM (coords from UPME when matched) ------------------
    enriched = xm.copy()
    # Replace coords on matched XM rows using UPME coords
    for j, (method, i, aux) in upme_idx_to_xm.items():
        # coordinates from this UPME site row
        u_lon = up_sites.at[j, "lon"]
        u_lat = up_sites.at[j, "lat"]
        enriched.at[i, "lon"] = u_lon
        enriched.at[i, "lat"] = u_lat
        # audit columns (optional, keep in enriched for transparency)
        enriched.at[i, "match_method"] = method
        enriched.at[i, "match_aux"] = aux
        enriched.at[i, "matched_with_upme_name"] = up_sites.at[j, "__name_upme"]

    # Validate: enriched coords must be present for every XM row
    if enriched["lon"].isna().any() or enriched["lat"].isna().any():
        # This should not happen if your sources are complete and parsed correctly.
        # We raise with details so you can inspect the culprit rows.
        bad = enriched.loc[enriched["lon"].isna() | enriched["lat"].isna(), [xm_name_col, "lon", "lat"]]
        raise RuntimeError(f"Enriched dataset contains NaN coords for {len(bad)} rows. "
                           f"Check coordinate parsing. Sample:\n{bad.head(10)}")


    # ------------------ Matched & Unmatched reporting ------------------
    matched_rows = []
    for j, (method, i, aux) in upme_idx_to_xm.items():
        matched_rows.append({
            "upme_index": j,
            "upme_name": up_sites.at[j, "__name_upme"],
            "upme_key": up_sites.at[j, "__key"],
            "xm_index": i,
            "xm_name": xm.at[i, "__name_xm"],
            "xm_key": xm.at[i, "__key"],
            "method": method,
            "distance_m": aux if method == "location" else np.nan,
            "score": aux if method.startswith("name") else np.nan,
        })
    df_matched = pd.DataFrame(matched_rows).sort_values(["method","upme_name"]).reset_index(drop=True)

    # # Unmatched UPME sites at record level (for transparency)
    # unmatched_upme_sites = up_sites.loc[[j for j in range(len(up_sites)) if j not in upme_idx_to_xm],
    #                                     ["__name_upme","__key","lon","lat"]].rename(columns={"__name_upme":"upme_name"})


    # ------------------ Unmatched UPME sites with reasons (diagnostics) ------------------
    NEARBY_KM = 10.0  # define "nearby" for diagnostics (not used for matching)
    NEARBY_M  = NEARBY_KM * 1000.0

    unmatched_idx = [j for j in range(len(up_sites)) if j not in upme_idx_to_xm]

    def _nearest_xm_for(lat, lon):
        best_i, best_d = None, float("inf")
        for i, xr in xm.iterrows():
            if pd.isna(xr["lon"]) or pd.isna(xr["lat"]):
                continue
            d = mu.haversine_m(lat, lon, xr["lat"], xr["lon"])
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    xm_key_set = set(xm["__key"].dropna().tolist())

    diag_rows = []
    for j in unmatched_idx:
        ur = up_sites.loc[j]
        uname = str(ur["__name_upme"])
        ukey  = ur["__key"]
        ulon  = ur["lon"]
        ulat  = ur["lat"]


        # all voltages in original UPME for this key
        voltages = sorted(up.loc[up["__key"] == ukey, "tension_kV"].dropna().unique().tolist())
        voltages_str = ";".join(str(v) for v in voltages) if voltages else None

        # defaults
        nearest_i = None
        nearest_d = np.nan
        nearest_name = None
        nearest_key  = None
        key_in_xm = (ukey in xm_key_set) if isinstance(ukey, str) and ukey else False



        if pd.isna(ulon) or pd.isna(ulat):
            reason = "upme_missing_coords"
            best_fuzzy_i = None
            best_fuzzy_name = None
            best_fuzzy_sc = np.nan
        else:
            # nearest-by-location for context
            nearest_i, nearest_d = _nearest_xm_for(ulat, ulon)
            if nearest_i is not None:
                nearest_name = str(xm.at[nearest_i, "__name_xm"])
                nearest_key  = xm.at[nearest_i, "__key"]

            # best fuzzy candidate (for explanation only)
            best_fuzzy_i, best_fuzzy_sc = None, -1.0
            best_fuzzy_name = None
            if DEFAULTS["use_fuzzy"]:
                for i, xr in xm.iterrows():
                    sc = mu.fuzzy_score(uname, str(xr["__name_xm"]))
                    if sc > best_fuzzy_sc:
                        best_fuzzy_sc, best_fuzzy_i = sc, i
                if best_fuzzy_i is not None:
                    best_fuzzy_name = str(xm.at[best_fuzzy_i, "__name_xm"])

            # decide reason
            if key_in_xm and (np.isnan(nearest_d) or nearest_d > radius_m):
                reason = "same_name_in_xm_but_outside_radius"
            elif (best_fuzzy_sc >= FUZZY_THRESHOLD) and (np.isnan(nearest_d) or nearest_d > radius_m):
                reason = "fuzzy_name_match_but_outside_radius"
            elif (not np.isnan(nearest_d)) and (nearest_d <= NEARBY_M):
                reason = "nearby_xm_but_name_mismatch"
            else:
                reason = "no_close_match"

        diag_rows.append({
            "upme_index": j,
            "upme_name": uname,
            "upme_key": ukey,
            "upme_lon": ulon,
            "upme_lat": ulat,
            "voltages_upme": voltages_str,
            "nearest_xm_index": nearest_i,
            "nearest_xm_name": nearest_name,
            "nearest_xm_key": nearest_key,
            "nearest_distance_m": nearest_d,
            "key_in_xm": key_in_xm,
            "best_fuzzy_xm_index": best_fuzzy_i,
            "best_fuzzy_xm_name": best_fuzzy_name,
            "best_fuzzy_score": best_fuzzy_sc if not np.isnan(best_fuzzy_sc) else np.nan,
            "reason": reason,
        })

    unmatched_upme_sites = pd.DataFrame(diag_rows).sort_values(["reason", "upme_name"]).reset_index(drop=True)

    # ---------- Append UPME-unmatched sites as new rows in the enriched XM dataset ----------

    # 1) Ensure unmatched has a 'voltages_upme' column. Your code already tries to fill it
    #    from up["tension_kV"]. If that column doesn't exist or the field is empty, try a robust fallback.
    if "voltages_upme" not in unmatched_upme_sites.columns or unmatched_upme_sites["voltages_upme"].isna().all():
        # find a voltage-like column in UPME
        VOLTAGE_COL_UPME = find_voltage_column(up)
        if VOLTAGE_COL_UPME is not None:
            volt_map = (
                up.groupby("__key", dropna=True)[VOLTAGE_COL_UPME]
                  .apply(lambda s: ";".join(sorted({str(v) for v in s.dropna()})))
                  .to_dict()
            )
            unmatched_upme_sites["voltages_upme"] = unmatched_upme_sites["upme_key"].map(volt_map).fillna("")
        else:
            unmatched_upme_sites["voltages_upme"] = ""

    # 2) Decide which column in XM holds "voltage"
    XM_VOLTAGE_COL = find_voltage_column(enriched)
    if XM_VOLTAGE_COL is None:
        # create a simple 'voltage' column if XM doesn't have one
        XM_VOLTAGE_COL = "voltage"
        if XM_VOLTAGE_COL not in enriched.columns:
            enriched[XM_VOLTAGE_COL] = np.nan

    # 3) Build rows to append using the SAME columns/order as 'enriched'
    append_rows = []
    for _, r in unmatched_upme_sites.iterrows():
        new_row = pd.Series(index=enriched.columns, dtype=object)

        # Name: use UPME cleaned name in the XM name column you detected earlier
        if xm_name_col in new_row.index:
            new_row[xm_name_col] = r["upme_name"]
        else:
            # fallback if schema changes
            new_row["name"] = r["upme_name"]

        # Coordinates from UPME unmatched site
        new_row["lon"] = r["upme_lon"]
        new_row["lat"] = r["upme_lat"]

        # Voltage: write UPME voltage list/string into the XM voltage column
        v_str = r.get("voltages_upme", "")
        new_row[XM_VOLTAGE_COL] = v_str if isinstance(v_str, str) and v_str.strip() else np.nan

        # Tag provenance
        new_row["match_method"] = "upme_only_append"
        new_row["match_aux"] = np.nan
        new_row["matched_with_upme_name"] = r["upme_name"]

        append_rows.append(new_row)

    if append_rows:
        append_df = pd.DataFrame(append_rows)
        append_df = append_df[enriched.columns]  # preserve column order
        enriched = pd.concat([enriched, append_df], ignore_index=True)

    num_appended = len(append_rows)
    print(f"UPME-only rows appended to enriched XM: {num_appended}")

    # ------------------ Save outputs ------------------
    out_enriched = args.out_enriched or "XM_enriched_with_UPME_coords.csv"
    out_matched  = args.out_matched or "UPME_to_XM_matched.csv"
    out_unmatched_sites = args.out_unmatched_sites or "UPME_unmatched_sites.csv"     # record-level (sites)


    # Drop helper columns from enriched before saving (keep audit if you like)
    save_cols = [c for c in enriched.columns if not c.startswith("__")]
    enriched[save_cols].to_csv(out_enriched, index=False, encoding="utf-8-sig")
    df_matched.to_csv(out_matched, index=False, encoding="utf-8-sig")
    unmatched_upme_sites.to_csv(out_unmatched_sites, index=False, encoding="utf-8-sig")


    # ------------------ Console report ------------------
    print("\n--- UPME XM Enrichment Report ---")
    print(f"XM rows: {len(xm)}")
    print(f"UPME rows: {len(up)} (sites used: {len(up_sites)})")
    print(f"Radius (m): {radius_m}")
    print()
    print(f"Matched UPME sites -- XM: {len(df_matched)}")
    print(f"Unmatched UPME sites (record-level): {len(unmatched_upme_sites)}")
    print()
    print("[OK] Saved files:")
    print(f"- {out_enriched}")
    print(f"- {out_matched}")
    print(f"- {out_unmatched_sites}")



if __name__ == "__main__":
    main()
