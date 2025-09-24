#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
osm_paratec_match.py

Cruza PARATEC (enriquecido con coords de UPME) contra OSM para producir:
  - OSM_PARATEC_enriched.csv
  - PARATEC_enriched_coords.csv
  - PARATEC_not_in_OSM.geojson
  - PARATEC_not_in_OSM.csv
  - PARATEC_not_in_OSM_missing_coords.csv
  - MATCHES_by_type.csv
  - MATCHES_summary.csv
  - (extra) OSM_not_in_PARATEC.csv
"""

import csv
import json
import pandas as pd

# --- Utilidades compartidas de matching ---
from matching_utils import (
    FUZZY_THRESHOLD, HAS_RAPIDFUZZ,
    normalize_core, tokenize, normalized_key,
    build_blocks, candidate_set, score_pair,
)

# -----------------------------
# Archivos de entrada/salida
# -----------------------------
PAR_CSV = "PARATEC_with_coords.csv"
OSM_CSV = "osm_substations_filtered.csv"

OUT_ENR                     = "OSM_PARATEC_enriched.csv"
OUT_PAR_ENR_COORDS          = "PARATEC_enriched_coords.csv"
OUT_PAR_NOT_GJ              = "PARATEC_not_in_OSM.geojson"
OUT_PAR_NOT_CSV             = "PARATEC_not_in_OSM.csv"
OUT_PAR_NOT_MISS_COORDS     = "PARATEC_not_in_OSM_missing_coords.csv"
OUT_MATCHES_BY_TYPE         = "MATCHES_by_type.csv"
OUT_MATCHES_SUMMARY         = "MATCHES_summary.csv"
OUT_OSM_NO_PAR              = "OSM_not_in_PARATEC.csv"   # extra

# -----------------------------
# Lectura robusta
# -----------------------------
def sniff_encoding_and_delim(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8"); enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1"); enc = "latin-1"
    sample = "\n".join(text.splitlines()[:50])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        delim = ","
    return enc, delim

def read_csv_smart(path: str):
    enc, delim = sniff_encoding_and_delim(path)
    df = pd.read_csv(
        path, sep=delim, dtype=str, keep_default_na=False, na_values=[""],
        encoding=enc, engine="python"
    )
    # limpia BOM y columnas Unnamed
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df, enc, delim

# -----------------------------
# GeoJSON helper (points)
# -----------------------------
def df_to_geojson_points(df: pd.DataFrame, out_path: str, lon_col="lon", lat_col="lat"):
    feats = []
    for _, r in df.iterrows():
        try:
            lon = float(str(r.get(lon_col, "")).strip())
            lat = float(str(r.get(lat_col, "")).strip())
        except Exception:
            continue
        props = {k: (None if pd.isna(v) else v) for k, v in r.items() if k not in (lon_col, lat_col)}
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props
        })
    gj = {"type": "FeatureCollection", "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)

# -----------------------------
# Main
# -----------------------------
def main():
    # Leer datasets
    df_par, par_enc, par_delim = read_csv_smart(PAR_CSV)
    df_osm, osm_enc, osm_delim = read_csv_smart(OSM_CSV)

    # Columnas de nombre
    par_name_col = "Nombre" if "Nombre" in df_par.columns else df_par.columns[0]
    osm_name_col = "name"   if "name"   in df_osm.columns else df_osm.columns[0]

    # Normalización y tokens para matching
    df_par["_key"]    = df_par[par_name_col].astype(str).map(normalized_key)
    df_par["_tokens"] = df_par[par_name_col].astype(str).map(normalize_core).map(tokenize)

    df_osm["_key"]    = df_osm[osm_name_col].astype(str).map(normalized_key)
    df_osm["_tokens"] = df_osm[osm_name_col].astype(str).map(normalize_core).map(tokenize)

    # OSM deduplicado por clave (node/way/rel -> una fila por key)
    df_osm_best = df_osm.drop_duplicates("_key", keep="first").copy()

    # Sets exactos
    keys_par = set(df_par["_key"])
    keys_osm = set(df_osm_best["_key"])
    exact_found   = keys_par & keys_osm
    exact_missing = keys_par - keys_osm

    # Índices para fuzzy (lado OSM)
    osm_keys_tokens = {k: toks for k, toks in df_osm_best[["_key", "_tokens"]].itertuples(index=False)}
    by_initial, by_lenband, token_index = build_blocks(osm_keys_tokens)

    if not HAS_RAPIDFUZZ:
        print("[WARN] rapidfuzz no está instalado; usando fallback con difflib "
              "(comparación sobre tokens ordenados). Recomendado: pip install rapidfuzz")

    # Fuzzy sobre PARATEC faltantes
    df_par_missing = df_par[df_par["_key"].isin(exact_missing)].copy()
    best_keys, best_scores = [], []
    for k, toks in df_par_missing[["_key","_tokens"]].itertuples(index=False):
        cands = candidate_set(k, toks, by_initial, by_lenband, token_index)
        if not cands:
            best_keys.append(None); best_scores.append(0.0); continue
        best_k, best_s = None, -1.0
        for ck in cands:
            s = score_pair(k, toks, ck, osm_keys_tokens[ck])
            if s > best_s:
                best_s, best_k = s, ck
        best_keys.append(best_k); best_scores.append(best_s)
    df_par_missing["best_osm_key"] = best_keys
    df_par_missing["best_score"]   = best_scores
    accepted = df_par_missing["best_score"] >= FUZZY_THRESHOLD

    # ------------------------------------
    # Construir mapeos exactos + fuzzy
    # ------------------------------------
    par_to_osm = {k: k for k in exact_found}
    # Agregar fuzzy aceptados
    for k, b, sc in df_par_missing[["_key","best_osm_key","best_score"]].itertuples(index=False):
        if pd.notna(b) and b and sc >= FUZZY_THRESHOLD:
            par_to_osm[k] = b
    osm_to_par = {v: k for k, v in par_to_osm.items() if v is not None}

    # ------------------------------------
    # OSM_PARATEC_enriched.csv
    # ------------------------------------
    # Prefijo PARATEC para no pisar columnas OSM
    par_cols = [c for c in df_par.columns if c not in ("_key","_tokens")]
    df_par_pref = df_par[["_key"] + par_cols].copy()
    df_par_pref = df_par_pref.add_prefix("PARATEC_").rename(columns={"PARATEC__key":"_par_key"})

    df_osm_enr = df_osm_best.copy()
    df_osm_enr["_par_key"] = df_osm_enr["_key"].map(osm_to_par)

    df_osm_enr = df_osm_enr.merge(
        df_par_pref, left_on="_par_key", right_on="_par_key", how="left"
    )

    df_osm_enr.to_csv(
        OUT_ENR, index=False, sep=osm_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # ------------------------------------
    # PARATEC_enriched_coords.csv
    # ------------------------------------
    # mapa de coords OSM por key
    osm_coord_map = df_osm_best.set_index("_key")[["lon","lat"]].to_dict(orient="index")

    df_par_ec = df_par.copy()
    if "lon" not in df_par_ec.columns: df_par_ec["lon"] = ""
    if "lat" not in df_par_ec.columns: df_par_ec["lat"] = ""

    def fill_par_coords(row):
        has_own = (str(row["lon"]).strip() != "") and (str(row["lat"]).strip() != "")
        if has_own:
            return row["lon"], row["lat"]
        k = row["_key"]
        k_osm = par_to_osm.get(k)
        if k_osm and k_osm in osm_coord_map:
            c = osm_coord_map[k_osm]
            return c.get("lon",""), c.get("lat","")
        return row["lon"], row["lat"]

    df_par_ec[["lon","lat"]] = df_par_ec.apply(lambda r: pd.Series(fill_par_coords(r)), axis=1)
    df_par_ec.to_csv(
        OUT_PAR_ENR_COORDS, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # ------------------------------------
    # PARATEC_not_in_OSM.* (CSV + GEOJSON + missing coords)
    # ------------------------------------
    par_not_keys = set(df_par["_key"]) - set(par_to_osm.keys())
    df_par_not = df_par[df_par["_key"].isin(par_not_keys)].copy()

    # CSV completo
    df_par_not.to_csv(
        OUT_PAR_NOT_CSV, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # GeoJSON usando coords del enriched_coords (mayor cobertura)
    df_to_geojson_points(df_par_ec[df_par_ec["_key"].isin(par_not_keys)],
                         OUT_PAR_NOT_GJ, lon_col="lon", lat_col="lat")

    # Subconjunto sin coords
    def _is_empty(v): 
        return (v is None) or (str(v).strip() == "") or pd.isna(v)
    mask_no_coords = df_par_ec["_key"].isin(par_not_keys) & (
        df_par_ec["lon"].map(_is_empty) | df_par_ec["lat"].map(_is_empty)
    )
    df_par_not_missing = df_par_ec.loc[mask_no_coords].copy()
    df_par_not_missing.to_csv(
        OUT_PAR_NOT_MISS_COORDS, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # ------------------------------------
    # OSM_not_in_PARATEC.csv  (extra útil)
    # ------------------------------------
    osm_not_keys = set(df_osm_best["_key"]) - set(osm_to_par.keys())
    df_osm_not = df_osm_best[df_osm_best["_key"].isin(osm_not_keys)].copy()

    # columnas mínimas amigables si existen
    cols_osm_min = []
    for c in ["name", "lon", "lat", "voltage", "operator", "substation", "osm_ids", "osm_types"]:
        if c in df_osm_not.columns:
            cols_osm_min.append(c)
    if not cols_osm_min:
        cols_osm_min = [osm_name_col]
        if "lon" in df_osm_not.columns and "lat" in df_osm_not.columns:
            cols_osm_min += ["lon","lat"]

    df_osm_not[cols_osm_min].to_csv(
        OUT_OSM_NO_PAR, index=False, sep=osm_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # ------------------------------------
    # MATCHES_by_type.csv  +  MATCHES_summary.csv
    # ------------------------------------
    exact_count = len(exact_found)
    fuzzy_count = int(accepted.sum())
    unmatched_count = len(par_not_keys)

    _df_bytype = pd.DataFrame([
        {"type": "exact",           "count": exact_count},
        {"type": "fuzzy_accepted",  "count": fuzzy_count},
        {"type": "unmatched",       "count": unmatched_count},
    ])
    _df_bytype.to_csv(
        OUT_MATCHES_BY_TYPE, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    total_par = len(keys_par)
    total_osm = len(keys_osm)
    total_matched = exact_count + fuzzy_count
    pct_par = 100.0 * total_matched / total_par if total_par else 0.0
    pct_osm = 100.0 * len(set(osm_to_par.keys())) / total_osm if total_osm else 0.0

    _df_summary = pd.DataFrame([{
        "par_total_unique": total_par,
        "osm_total_unique": total_osm,
        "exact": exact_count,
        "fuzzy_accepted": fuzzy_count,
        "matched_total": total_matched,
        "par_not_in_osm": unmatched_count,
        "osm_not_in_par": len(osm_not_keys),
        "pct_par_matched": round(pct_par, 1),
        "pct_osm_covered": round(pct_osm, 1),
    }])
    _df_summary.to_csv(
        OUT_MATCHES_SUMMARY, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # ------------------------------------
    # Consola: resumen y porcentajes
    # ------------------------------------
    matched_par_keys = set(par_to_osm.keys())
    matched_osm_keys = set(osm_to_par.keys())

    print("--- Summary ---")
    print(f"PARATEC filas:                 {len(df_par)}")
    print(f"PARATEC únicas por key:        {len(keys_par)}")
    print(f"OSM filas (filtradas):         {len(df_osm)}")
    print(f"OSM únicas por key:            {len(keys_osm)}")
    print(f"Exact matches:                 {exact_count}")
    print(f"Fuzzy aceptados (>= {FUZZY_THRESHOLD}): {fuzzy_count}")
    print(f"PARATEC no en OSM:             {len(par_not_keys)} (ver {OUT_PAR_NOT_CSV})")
    print(f"OSM no en PARATEC:             {len(osm_not_keys)} (ver {OUT_OSM_NO_PAR})")
    print(f"Matched con XM (PARATEC):      {len(matched_par_keys)} / {len(keys_par)} ({pct_par:.1f}%)")
    print(f"OSM cubierto por XM:           {len(matched_osm_keys)} / {len(keys_osm)} ({pct_osm:.1f}%)")
    print(f"Salida enriquecida:            {OUT_ENR}")

if __name__ == "__main__":
    main()
