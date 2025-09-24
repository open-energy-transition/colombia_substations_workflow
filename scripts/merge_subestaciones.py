#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_subestaciones.py

Generates:
  - PARATEC_with_coords.csv (PARATEC + lon/lat from UPME by exact + fuzzy match)
  - PARATEC_unmatched_in_UPME.csv (PARATEC substations without an accepted match)
  - PARATEC_fuzzy_matches.csv (substations matched by fuzzy logic, with score and coords)

Features:
  - Robust reading (auto-detects encoding and delimiter; cleans BOM/Unnamed).
  - Aggressive normalization and tokenization with Roman/Arabic numerals and domain stopwords.
  - Exact + fuzzy matching (RapidFuzz if available; difflib fallback on token-sorted strings).
  - UPME deduplication prioritizing rows with valid lon/lat.
  - Outputs with the same delimiter as PARATEC and UTF-8 with BOM.
  - ASCII-safe prints for Windows console.
"""

import sys
import csv
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Shared matching utilities (centralized) ---
from matching_utils import (
    FUZZY_THRESHOLD, HAS_RAPIDFUZZ,
    strip_accents, normalize_core, tokenize,
    normalized_key,  # strict key (backward-compatible with your pipeline)
    build_blocks, candidate_set, score_pair,
)

# -----------------------------
# Configuración
# -----------------------------
PARA_CSV = "PARATEC_substations.csv"
UPME_CSV = "subestaciones_upme.csv"
OUT_ENR  = "PARATEC_with_coords.csv"
OUT_UNM  = "PARATEC_unmatched_in_UPME.csv"
OUT_FZY  = "PARATEC_fuzzy_matches.csv"

# Columnas preferidas (si existen en PARATEC) para incluir en el reporte de no encontrados
PAR_COLS_REPORT_PREF = [
    "Voltaje nominal de operación [kV]",
    "Departamento",
    "Municipio",
    "Subárea operativa",
]

# -----------------------------
# Utilidades de lectura/escritura
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
        path,
        sep=delim,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding=enc,
        engine="python",
    )
    # Limpia BOM y columnas Unnamed
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df, enc, delim

# -----------------------------
# Scoring y matching (envolturas locales mínimas)
# -----------------------------
def build_blocks_local(keys_tokens: dict):
    """Envoltura por compatibilidad: reutiliza build_blocks del utils."""
    return build_blocks(keys_tokens)

def candidate_set_local(k, toks, by_initial, by_lenband, token_index):
    """Envoltura por compatibilidad: reutiliza candidate_set del utils."""
    return candidate_set(k, toks, by_initial, by_lenband, token_index)

def score_pair_local(a_key, a_tokens, b_key, b_tokens):
    """Envoltura por compatibilidad: reutiliza score_pair del utils."""
    return score_pair(a_key, a_tokens, b_key, b_tokens)

def best_match_for(key, toks, upme_keys_tokens, blocks):
    by_initial, by_lenband, token_index = blocks
    cands = candidate_set_local(key, toks, by_initial, by_lenband, token_index)
    if not cands:
        return None, 0.0
    best_k, best_s = None, -1.0
    for ck in cands:
        s = score_pair_local(key, toks, ck, upme_keys_tokens[ck])
        if s > best_s:
            best_s, best_k = s, ck
    return best_k, best_s

# -----------------------------
# Auxiliares
# -----------------------------
def to_float(x):
    if x is None or x == "":
        return np.nan
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

# -----------------------------
# Pipeline
# -----------------------------
def main():
    df_par, par_enc, par_delim = read_csv_smart(PARA_CSV)
    df_upm, _, _               = read_csv_smart(UPME_CSV)

    # Columna de nombre real en cada dataset
    par_name_col = "Nombre" if "Nombre" in df_par.columns else df_par.columns[0]
    upm_name_col = "nombre_subestacion" if "nombre_subestacion" in df_upm.columns else df_upm.columns[0]

    # Normalización y tokens
    df_par["_key"]    = df_par[par_name_col].astype(str).map(normalized_key)
    df_par["_tokens"] = df_par[par_name_col].astype(str).map(normalize_core).map(tokenize)

    df_upm["_key"]    = df_upm[upm_name_col].astype(str).map(normalized_key)
    df_upm["_tokens"] = df_upm[upm_name_col].astype(str).map(normalize_core).map(tokenize)

    # lon/lat numéricos
    if "lon" in df_upm.columns: df_upm["lon"] = df_upm["lon"].map(to_float)
    if "lat" in df_upm.columns: df_upm["lat"] = df_upm["lat"].map(to_float)

    # Deduplicar UPME priorizando lon/lat
    df_upm["_rank"] = df_upm.apply(lambda r: int(pd.notna(r.get("lon")) and pd.notna(r.get("lat"))), axis=1)
    df_upm_best = df_upm.sort_values(["_key","_rank"], ascending=[True, False]).drop_duplicates("_key", keep="first")

    # Exactos
    keys_par  = set(df_par["_key"])
    keys_upme = set(df_upm_best["_key"])
    exact_found   = keys_par & keys_upme
    exact_missing = keys_par - keys_upme

    # Índices para fuzzy
    upme_keys_tokens = {k: toks for k, toks in df_upm_best[["_key","_tokens"]].itertuples(index=False)}
    blocks = build_blocks_local(upme_keys_tokens)

    if not HAS_RAPIDFUZZ:
        print("[WARN] rapidfuzz no está instalado; usando fallback con difflib "
              "(comparación sobre tokens ordenados). Recomendado: pip install rapidfuzz")

    # Fuzzy para los faltantes
    df_par_missing = df_par[df_par["_key"].isin(exact_missing)].copy()
    best_keys, best_scores = [], []
    for k, toks in df_par_missing[["_key","_tokens"]].itertuples(index=False):
        bk, sc = best_match_for(k, toks, upme_keys_tokens, blocks)
        best_keys.append(bk); best_scores.append(sc)
    df_par_missing["best_upme_key"] = best_keys
    df_par_missing["best_score"]    = best_scores
    accepted = df_par_missing["best_score"] >= FUZZY_THRESHOLD

    # Mapa coordenadas UPME
    coord_map = df_upm_best.set_index("_key")[["lon","lat"]].to_dict(orient="index")

    # Enriquecido: mantener orden de PARATEC y añadir lon/lat al final
    base_cols = [c for c in df_par.columns if c not in ("_key","_tokens")]
    for c in ("lon","lat"):
        if c not in base_cols: base_cols.append(c)

    df_par_enr = df_par.copy()
    # Exactos
    df_par_enr["lon"] = df_par_enr["_key"].map(lambda k: coord_map.get(k, {}).get("lon"))
    df_par_enr["lat"] = df_par_enr["_key"].map(lambda k: coord_map.get(k, {}).get("lat"))
    # Fuzzy aceptados (solo rellena donde haga falta)
    acc_map = dict(zip(df_par_missing.loc[accepted, "_key"], df_par_missing.loc[accepted, "best_upme_key"]))
    idxs = df_par_enr["_key"].isin(acc_map.keys())
    if idxs.any():
        def fill_from_acc(row):
            if pd.notna(row["lon"]) and pd.notna(row["lat"]):
                return row["lon"], row["lat"]
            bk = acc_map.get(row["_key"])
            if bk and bk in coord_map:
                return coord_map[bk]["lon"], coord_map[bk]["lat"]
            return row["lon"], row["lat"]
        filled = [fill_from_acc(r) for _, r in df_par_enr.loc[idxs, ["_key","lon","lat"]].iterrows()]
        df_par_enr.loc[idxs, "lon"] = [x[0] for x in filled]
        df_par_enr.loc[idxs, "lat"] = [x[1] for x in filled]

    # Formato lon/lat
    for c in ("lon","lat"):
        if c in df_par_enr.columns:
            df_par_enr[c] = df_par_enr[c].apply(lambda v: (f"{float(v):.6f}" if pd.notna(v) else ""))

    # Guardar enriquecido
    df_par_enr[base_cols].to_csv(
        OUT_ENR, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # -----------------------------
    # Reporte de fuzzy aceptados
    # -----------------------------
    # accepted: filas en df_par_missing que superaron el umbral
    df_fuzzy_acc = df_par_missing.loc[accepted, ["_key", "best_upme_key", "best_score"]].copy()

    # Traer nombres originales PARATEC y UPME, y coords finales usadas
    df_par_names = df_par[["_key", par_name_col]].drop_duplicates("_key").rename(columns={par_name_col: "PARATEC_Nombre"})
    df_upm_names = df_upm_best[["_key", upm_name_col, "lon", "lat"]].rename(columns={
        "_key": "upme_key",
        upm_name_col: "UPME_Nombre"
    })

    df_fuzzy_acc = df_fuzzy_acc.merge(df_par_names, on="_key", how="left")
    df_fuzzy_acc = df_fuzzy_acc.merge(df_upm_names, left_on="best_upme_key", right_on="upme_key", how="left")

    # Columnas finales del reporte fuzzy
    cols_fuzzy = ["PARATEC_Nombre", "UPME_Nombre", "best_score", "lon", "lat"]
    df_fuzzy_out = df_fuzzy_acc[cols_fuzzy].copy().sort_values(by="best_score", ascending=False)

    df_fuzzy_out.to_csv(
        OUT_FZY, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # -----------------------------
    # Reporte de no encontrados
    # -----------------------------
    df_unmatched_keys = df_par_missing.loc[~accepted, ["_key", "best_score"]].copy()

    extra_cols = [c for c in PAR_COLS_REPORT_PREF if c in df_par.columns]
    cols_from_par = ["_key", par_name_col] + [c for c in extra_cols if c != par_name_col]
    df_par_slice = df_par[cols_from_par].drop_duplicates("_key")

    report = df_unmatched_keys.merge(df_par_slice, on="_key", how="left")
    report = report.rename(columns={par_name_col: "PARATEC_Nombre"})

    cols_final = ["PARATEC_Nombre"] + [c for c in extra_cols if c != par_name_col] + ["best_score"]
    cols_final = [c for c in cols_final if c in report.columns]
    report = report[cols_final].drop_duplicates().sort_values(by="PARATEC_Nombre", na_position="last")

    report.to_csv(
        OUT_UNM, index=False, sep=par_delim, encoding="utf-8-sig",
        lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL
    )

    # Resumen ASCII-safe
    keys_par  = set(df_par["_key"])
    keys_upme = set(df_upm_best["_key"])

    print(f"Total PARATEC (unicas): {len(keys_par)}")
    print(f"Total UPME (unicas):    {len(keys_upme)}")
    print(f"Exact matches:          {len(keys_par & keys_upme)}")
    print(f"Fuzzy aceptados (>= {FUZZY_THRESHOLD}): {int(accepted.sum())}")
    print(f"No encontrados (reporte): {len(report)}")
    print(f"Salida enriquecida: {OUT_ENR}")
    print(f"Reporte no encontrados: {OUT_UNM}")
    print(f"Reporte fuzzy aceptados: {OUT_FZY}")  # linea adicional

if __name__ == "__main__":
    main()
