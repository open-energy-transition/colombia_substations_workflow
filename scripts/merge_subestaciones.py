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
  - Exact + fuzzy matching (rapidfuzz if available; difflib as fallback).
  - UPME deduplication prioritizing rows with valid lon/lat.
  - Outputs with the same delimiter as PARATEC and UTF-8 with BOM.
  - ASCII-safe prints for Windows console.
"""

import sys
import csv
import re
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict

try:
    from rapidfuzz import fuzz
    try:
        from rapidfuzz.distance import JaroWinkler as RF_JW
    except Exception:
        RF_JW = None
    HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    RF_JW = None
    HAS_RAPIDFUZZ = False

# -----------------------------
# Configuración
# -----------------------------
PARA_CSV = "PARATEC_substations.csv"
UPME_CSV = "subestaciones_upme.csv"
OUT_ENR  = "PARATEC_with_coords.csv"
OUT_UNM  = "PARATEC_unmatched_in_UPME.csv"
OUT_FZY  = "PARATEC_fuzzy_matches.csv"

FUZZY_THRESHOLD = 63

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
# Normalización y tokenización
# -----------------------------
_ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
}
_STOPWORDS = {
    "subestacion","subestación","se","s/e","estacion","estación",
    "san","santo","santa","sta","sto","sa","calle","cll","av","avenida",
    "norte","sur","este","oeste","oriente","occidente","de","del","la","el",
    "eeb","eeeb","bogota","bogotá"
}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def normalize_core(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = strip_accents(name).lower().strip()
    s = re.sub(r"\(.*?\)", "", s)                     # quita alias (... )
    s = re.sub(r"\b\d+(\.\d+)?\s*kv\b", "", s)        # quita "115 kV"
    s = re.sub(r"\s+", " ", s).strip()
    return s

def roman_to_arabic_token(tok: str) -> str:
    return _ROMAN_MAP.get(tok, tok)

def arabic_to_roman_token(tok: str) -> str:
    inv = {v: k for k, v in _ROMAN_MAP.items()}
    return inv.get(tok, tok)

def tokenize(name: str):
    s = re.sub(r"[^a-z0-9\s]+", " ", name)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        t1 = roman_to_arabic_token(t); out.append(t1)
        t2 = arabic_to_roman_token(t1)
        if t2 != t1: out.append(t2)
    out = [t for t in out if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]
    return out

def normalized_key(name: str) -> str:
    core = normalize_core(name)
    core = re.sub(r"[^a-z0-9]+", "", core)
    for r, a in _ROMAN_MAP.items():
        core = re.sub(rf"{r}(?![a-z0-9])", a, core)
    return core

# -----------------------------
# Scoring y matching
# -----------------------------
def build_blocks(keys_tokens: dict):
    by_initial = defaultdict(set)
    by_lenband = defaultdict(set)
    token_index = defaultdict(set)
    for k, toks in keys_tokens.items():
        if not k: continue
        by_initial[k[0]].add(k)
        by_lenband[len(k)//3].add(k)
        for t in set(toks):
            token_index[t].add(k)
    return by_initial, by_lenband, token_index

def candidate_set(k, toks, by_initial, by_lenband, token_index):
    cands = set()
    if k:
        cands |= by_initial.get(k[0], set())
        cands |= by_lenband.get(len(k)//3, set())
    for t in set(toks):
        cands |= token_index.get(t, set())
    return list(cands)

def score_pair(a_key, a_tokens, b_key, b_tokens, rf=None):
    sa, sb = set(a_tokens), set(b_tokens)
    jacc = 100.0 * (len(sa & sb) / len(sa | sb)) if (sa or sb) else 0.0
    tset = tsort = part = 0.0; jw = 0.0

    #RapidFuzz importado a nivel módulo
    if HAS_RAPIDFUZZ and fuzz is not None:
        tset  = fuzz.token_set_ratio(a_key, b_key)
        tsort = fuzz.token_sort_ratio(a_key, b_key)
        part  = fuzz.partial_ratio(a_key, b_key)
        jw    = 100.0 * RF_JW.normalized_similarity(a_key, b_key) if RF_JW else 0.0
    else:
        # Fallback  sin RapidFuzz: comparar sobre tokens ORDENADOS
        import difflib
        a_tok = " ".join(sorted(a_tokens))
        b_tok = " ".join(sorted(b_tokens))
        ratio = 100.0 * difflib.SequenceMatcher(None, a_tok, b_tok).ratio()
        tset = tsort = part = ratio
        jw   = 0.0

    return 0.35*tset + 0.25*tsort + 0.15*part + 0.15*jacc + 0.10*jw

def best_match_for(key, toks, upme_keys_tokens, blocks, rf=None):
    by_initial, by_lenband, token_index = blocks
    cands = candidate_set(key, toks, by_initial, by_lenband, token_index)
    if not cands:
        return None, 0.0
    best_k, best_s = None, -1.0
    for ck in cands:
        s = score_pair(key, toks, ck, upme_keys_tokens[ck])
        if s > best_s:
            best_s, best_k = s, ck
    return best_k, best_s

# -----------------------------
# Auxiliares
# -----------------------------
def to_float(x):
    if x is None or x == "": return np.nan
    s = str(x).strip().replace(",", ".")
    try: return float(s)
    except ValueError: return np.nan

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
    blocks = build_blocks(upme_keys_tokens)


    if not HAS_RAPIDFUZZ:
        print("[WARN] rapidfuzz no está instalado; usando fallback con difflib. "
          "Recomendado: pip install rapidfuzz para mayor precisión y rendimiento.")
    

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
    # Fuzzy aceptados
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
