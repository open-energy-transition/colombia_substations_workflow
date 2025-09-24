#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
osm_paratec_match.py

Cruza PARATEC (enriquecido con coords de UPME) contra OSM para:
  - Generar OSM_PARATEC_enriched.csv (OSM con atributos de PARATEC donde hay match).
  - Reportar PARATEC_no_en_OSM.csv (PARATEC sin match en OSM).
  - Reportar OSM_not_in_PARATEC.csv (OSM sin match en PARATEC).
  - Imprimir porcentajes de cobertura (XM cubierto en OSM y OSM cubierto por XM).

Requisitos previos:
  - Ejecutar merge_subestaciones.py para producir PARATEC_with_coords.csv
  - Ejecutar get_osm_subs.py para producir osm_substations_filtered.csv

Dependencias de matching centralizadas en matching_utils.py
"""

import csv
import re
import json
import pandas as pd
from typing import List, Dict

# --- Utilidades compartidas de matching ---
from matching_utils import (
    FUZZY_THRESHOLD, HAS_RAPIDFUZZ,
    normalize_core, tokenize,
    normalized_key,            # clave estricta (compatibilidad hacia atrás)
    build_blocks, candidate_set, score_pair,
)

# -----------------------------
# Configuración de I/O
# -----------------------------
PAR_CSV   = "PARATEC_with_coords.csv"
OSM_CSV   = "osm_substations_filtered.csv"

OUT_ENR   = "OSM_PARATEC_enriched.csv"
OUT_PAR_NO_OSM = "PARATEC_no_en_OSM.csv"
OUT_OSM_NO_PAR = "OSM_not_in_PARATEC.csv"

# -----------------------------
# Lectura robusta
# -----------------------------
def sniff_encoding_and_delim(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        enc = "latin-1"
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
# Lógica principal
# -----------------------------
def main():
    # Leer datasets
    df_par, par_enc, par_delim = read_csv_smart(PAR_CSV)
    df_osm, osm_enc, osm_delim = read_csv_smart(OSM_CSV)

    # Columnas de nombre y coords
    par_name_col = "Nombre" if "Nombre" in df_par.columns else df_par.columns[0]
    osm_name_col = "name"   if "name"   in df_osm.columns else df_osm.columns[0]

    # Normalización/trabajo de claves
    df_par["_key"]    = df_par[par_name_col].astype(str).map(normalized_key)
    df_par["_tokens"] = df_par[par_name_col].astype(str).map(normalize_core).map(tokenize)

    df_osm["_key"]    = df_osm[osm_name_col].astype(str).map(normalized_key)
    df_osm["_tokens"] = df_osm[osm_name_col].astype(str).map(normalize_core).map(tokenize)

    # OSM deduplicado por clave (node/way/rel a una sola fila por nombre normalizado)
    df_osm_best = df_osm.drop_duplicates("_key", keep="first").copy()

    # Conjuntos para exact
    keys_par = set(df_par["_key"])
    keys_osm = set(df_osm_best["_key"])

    exact_found   = keys_par & keys_osm
    exact_missing = keys_par - keys_osm

    # Índices para fuzzy (solo candidatos en OSM)
    osm_keys_tokens = {k: toks for k, toks in df_osm_best[["_key", "_tokens"]].itertuples(index=False)}
    blocks = build_blocks(osm_keys_tokens)

    if not HAS_RAPIDFUZZ:
        print("[WARN] rapidfuzz no está instalado; usando fallback con difflib "
              "(comparación sobre tokens ordenados). Recomendado: pip install rapidfuzz")

    # Fuzzy sobre PARATEC faltantes
    df_par_missing = df_par[df_par["_key"].isin(exact_missing)].copy()
    best_keys, best_scores = [], []
    for k, toks in df_par_missing[["_key", "_tokens"]].itertuples(index=False):
        by_initial, by_lenband, token_index = blocks
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

    # -----------------------------
    # Construir enriquecido OSM<-PARATEC
    # -----------------------------
    # Mapa de key PARATEC -> key OSM (exact + fuzzy aceptado)
    par_to_osm = {k: k for k in exact_found}
    par_to_osm.update(dict(zip(
        df_par_missing.loc[accepted, "_key"], 
        df_par_missing.loc[accepted, "best_osm_key"]
    )))
    # Inverso OSM -> PARATEC
    osm_to_par = {v: k for k, v in par_to_osm.items() if v is not None}

    # Preparar PARATEC con prefijo para no pisar columnas OSM
    par_cols = [c for c in df_par.columns if c not in ("_key", "_tokens")]
    df_par_pref = df_par[["_key"] + par_cols].copy()
    df_par_pref = df_par_pref.add_prefix("PARATEC_").rename(columns={"PARATEC__key":"_par_key"})

    # OSM base (únicos)
    df_osm_enr = df_osm_best.copy()
    df_osm_enr["_par_key"] = df_osm_enr["_key"].map(osm_to_par)

    # Join atributos PARATEC a OSM por _par_key
    df_osm_enr = df_osm_enr.merge(
        df_par_pref, left_on="_par_key", right_on="_par_key", how="left"
    )

    # Guardar enriquecido (mantén columnas originales + PARATEC_* a la derecha)
    df_osm_enr.to_csv(
        OUT_ENR, index=False, sep=osm_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # -----------------------------
    # Reporte PARATEC no cubierto por OSM
    # -----------------------------
    par_not_keys = set(df_par["_key"]) - set(par_to_osm.keys())
    df_par_not = df_par[df_par["_key"].isin(par_not_keys)].copy()

    # Export sencillo a CSV con lo más útil
    cols_par_min = []
    for c in ["Nombre", "lon", "lat", "Voltaje nominal de operación [kV]", "Departamento", "Municipio"]:
        if c in df_par_not.columns:
            cols_par_min.append(c)
    if not cols_par_min:
        cols_par_min = [par_name_col]
    df_par_not[cols_par_min].to_csv(
        OUT_PAR_NO_OSM, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # -----------------------------
    # Reporte OSM no cubierto por PARATEC
    # -----------------------------
    osm_not_keys = set(df_osm_best["_key"]) - set(osm_to_par.keys())
    df_osm_not = df_osm_best[df_osm_best["_key"].isin(osm_not_keys)].copy()

    cols_osm_min = []
    for c in ["name", "lon", "lat", "voltage", "operator", "substation"]:
        if c in df_osm_not.columns:
            cols_osm_min.append(c)
    if not cols_osm_min:
        cols_osm_min = [osm_name_col, "lon", "lat"] if {"lon","lat"}.issubset(df_osm_not.columns) else [osm_name_col]

    df_osm_not[cols_osm_min].to_csv(
        OUT_OSM_NO_PAR, index=False, sep=osm_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # -----------------------------
    # Consola: resumen y porcentajes
    # -----------------------------
    matched_par_keys = set(par_to_osm.keys())
    matched_osm_keys = set(osm_to_par.keys())

    print("--- Summary ---")
    print(f"PARATEC filas:                 {len(df_par)}")
    print(f"PARATEC únicas por key:        {len(keys_par)}")
    print(f"OSM filas (filtradas):         {len(df_osm)}")
    print(f"OSM únicas por key:            {len(keys_osm)}")
    print(f"Exact matches:                 {len(exact_found)}")
    print(f"Fuzzy aceptados (>= {FUZZY_THRESHOLD}): {int(accepted.sum())}")
    print(f"PARATEC no en OSM:             {len(par_not_keys)} (ver {OUT_PAR_NO_OSM})")
    print(f"OSM no en PARATEC:             {len(osm_not_keys)} (ver {OUT_OSM_NO_PAR})")

    # % de XM cubierto por OSM (completitud de OSM respecto a XM)
    pct_par = 100.0 * len(matched_par_keys) / len(keys_par) if keys_par else 0.0
    print(f"Matched con XM (PARATEC):      {len(matched_par_keys)} / {len(keys_par)} ({pct_par:.1f}%)")

    # % de OSM cubierto por XM (concordancia de OSM con XM)
    pct_osm = 100.0 * len(matched_osm_keys) / len(keys_osm) if keys_osm else 0.0
    print(f"OSM cubierto por XM:           {len(matched_osm_keys)} / {len(keys_osm)} ({pct_osm:.1f}%)")

    # Salida principal para mapa
    print(f"Salida enriquecida:            {OUT_ENR}")

if __name__ == "__main__":
    main()
