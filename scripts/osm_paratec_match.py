#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, re, unicodedata, math, json
from collections import defaultdict
import pandas as pd
import numpy as np

# ---- Usamos utils para normalización y scoring (no cambiamos lógica de outputs) ----
from matching_utils import (
    normalized_key, tokenize, score_pair,
)

# ---------------- Config ----------------
PARA_CSV = "PARATEC_with_coords.csv"
OSM_CSV  = "osm_substations_filtered.csv"

FUZZY_THRESHOLD = 70

OUT_PAR_ENR   = "PARATEC_enriched_coords.csv"
OUT_PAR_GJ    = "PARATEC_not_in_OSM.geojson"
OUT_PAR_MISS  = "PARATEC_not_in_OSM_missing_coords.csv"
OUT_MATCH_SUM = "MATCHES_summary.csv"

OUT_OSM_ENR_MIN = "OSM_PARATEC_enriched.csv"
OUT_PAR_NOT_CSV = "PARATEC_not_in_OSM.csv"
OUT_MATCH_TYPE  = "MATCHES_by_type.csv"

OUT_OSM_NOT     = "OSM_not_in_PARATEC.csv"
OUT_OSM_NOT_GJ  = "OSM_not_in_PARATEC.geojson"


# ---------------- Helpers IO (respeta tu forma de leer/escribir) ----------------
def sniff_csv_meta(path):
    # encoding
    enc = "utf-8"
    raw = open(path, "rb").read(4096)
    if raw.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    else:
        try:
            raw.decode("utf-8")
            enc = "utf-8"
        except UnicodeDecodeError:
            enc = "latin-1"
    # delimiter
    with open(path, "r", encoding=enc, newline="") as f:
        sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    return delim, enc, "\n"


def read_csv_smart(path):
    delim, enc, eol = sniff_csv_meta(path)
    df = pd.read_csv(path, encoding=enc, sep=delim, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    return df, (delim, enc, eol)


def to_csv_like_source(df, path_out, like_path):
    delim, enc, eol = sniff_csv_meta(like_path)
    df.to_csv(path_out, index=False, encoding=enc, sep=delim, line_terminator=eol)


# ---------------- Normalización (wrapper a utils) ----------------
def norm_name_key(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    return normalized_key(s) if s else ""


# ---------------- Wrappers seguros para bloques y candidatos ----------------
def build_blocks_safe(series: pd.Series):
    """
    Construye bloques por tokens de forma segura forzando string normalizado.
    Evita keys no indexables/numéricas que rompen utils.
    """
    blocks = defaultdict(set)
    for idx, val in series.items():
        key = norm_name_key(val)
        if not key:
            continue
        for t in tokenize(key):
            if t:
                blocks[t].add(idx)
    return blocks


def candidate_set_safe(name: str, blocks: dict, df: pd.DataFrame, col: str, max_cands=200):
    """
    Genera candidatos por intersección de tokens, con guardas para strings.
    """
    key = norm_name_key(name)
    if not key:
        return set()
    toks = [t for t in tokenize(key) if t]
    if not toks:
        return set()

    cand_sets = []
    for t in toks:
        s = blocks.get(t)
        if s:
            cand_sets.append(s)
    if not cand_sets:
        return set()

    cands = set.intersection(*cand_sets) if cand_sets else set()

    # Recorte por longitud similar (idéntica heurística a la versión anterior)
    if len(cands) > max_cands:
        L = len(key)
        trimmed = set()
        for i in cands:
            k2 = norm_name_key(df.at[i, col])
            if abs(len(k2) - L) <= 10:
                trimmed.add(i)
        cands = trimmed or cands
    return cands


# ---------------- Carga datos ----------------
def load_paratec():
    df, meta = read_csv_smart(PARA_CSV)
    for c in ["Nombre", "lat", "lon"]:
        if c not in df.columns:
            df[c] = ""
    return df, meta


def load_osm():
    df, meta = read_csv_smart(OSM_CSV)
    for c in ["name", "lat", "lon"]:
        if c not in df.columns:
            df[c] = ""
    return df, meta


# ---------------- Matching principal (exact + fuzzy) ----------------
def run_matching(paratec_df: pd.DataFrame, osm_df: pd.DataFrame):
    paratec_df["_key"] = paratec_df["Nombre"].map(norm_name_key)
    osm_df["_key"]     = osm_df["name"].map(norm_name_key)

    # Exact por _key
    osm_map = {k: i for i, k in osm_df["_key"].items() if k}
    exact_hits, no_hits = [], []
    for i, k in paratec_df["_key"].items():
        if k and k in osm_map:
            exact_hits.append((i, osm_map[k]))
        else:
            no_hits.append(i)

    print(f"[INFO] Exact matches: {len(exact_hits)} | Pending fuzzy: {len(no_hits)}")

    # Fuzzy por bloques (wrappers seguros)
    blocks = build_blocks_safe(osm_df["name"])
    fuzzy_hits = []
    for i in no_hits:
        name = paratec_df.at[i, "Nombre"]
        cand_idx = candidate_set_safe(name, blocks, osm_df, "name", max_cands=200)
        if not cand_idx:
            continue
        key_a = norm_name_key(name)
        best_j, best_score = None, -1.0
        for j in cand_idx:
            s = score_pair(key_a, norm_name_key(osm_df.at[j, "name"]))
            if s > best_score:
                best_score, best_j = s, j
        if best_j is not None and best_score >= FUZZY_THRESHOLD:
            fuzzy_hits.append((i, best_j, best_score))

    print(f"[INFO] Fuzzy matches (>= {FUZZY_THRESHOLD}): {len(fuzzy_hits)}")
    return exact_hits, fuzzy_hits


# ---------------- Utilidad para únicos por _key (para console summary) ----------------
def pick_best_by_key(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    def _valid_xy(r):
        try:
            lat = float(r.get("lat", "") or "nan")
            lon = float(r.get("lon", "") or "nan")
            return math.isfinite(lat) and math.isfinite(lon)
        except Exception:
            return False
    out = []
    for k, g in df.groupby(key_col, dropna=False):
        if not k:
            out.append(g.iloc[0]); continue
        ok = g[g.apply(_valid_xy, axis=1)]
        out.append(ok.iloc[0] if len(ok) else g.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


# ---------------- Enriquecimiento y salidas (idénticos a los tuyos) ----------------
def enrich_and_write_outputs(paratec_df, osm_df, exact_hits, fuzzy_hits, par_meta, osm_meta):
    # 1) PARATEC_enriched_coords.csv
    enr = paratec_df.copy()
    enr["match_type"]  = ""
    enr["match_name"]  = ""
    enr["match_score"] = ""
    for i, j in exact_hits:
        enr.at[i, "lat"] = osm_df.at[j, "lat"]; enr.at[i, "lon"] = osm_df.at[j, "lon"]
        enr.at[i, "match_type"] = "exact"; enr.at[i, "match_name"] = osm_df.at[j, "name"]; enr.at[i, "match_score"] = ""
    for i, j, s in fuzzy_hits:
        enr.at[i, "lat"] = osm_df.at[j, "lat"]; enr.at[i, "lon"] = osm_df.at[j, "lon"]
        enr.at[i, "match_type"] = "fuzzy"; enr.at[i, "match_name"] = osm_df.at[j, "name"]; enr.at[i, "match_score"] = f"{s:.0f}"
    to_csv_like_source(enr, OUT_PAR_ENR, PARA_CSV)
    print(f"[OK] Wrote {OUT_PAR_ENR} ({len(enr)} rows)")

    # 2) PARATEC_not_in_OSM.csv
    matched_i = {i for i, _ in exact_hits} | {i for i, _, _ in fuzzy_hits}
    par_not = paratec_df.loc[~paratec_df.index.isin(matched_i)].copy()
    to_csv_like_source(par_not, OUT_PAR_NOT_CSV, PARA_CSV)
    print(f"[OK] Wrote {OUT_PAR_NOT_CSV} ({len(par_not)} rows)")

    # 3) PARATEC_not_in_OSM.geojson
    feats = []
    for _, r in par_not.iterrows():
        try:
            lat = float(r.get("lat", "") or "nan"); lon = float(r.get("lon", "") or "nan")
        except Exception:
            lat, lon = float("nan"), float("nan")
        if not (math.isfinite(lat) and math.isfinite(lon)): continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"Nombre": r.get("Nombre", ""), "source": "PARATEC"},
        })
    with open(OUT_PAR_GJ, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {OUT_PAR_GJ} ({len(feats)} points)")

    # 4) PARATEC_not_in_OSM_missing_coords.csv
    def _hasxy(r):
        try:
            lat = float(r.get("lat", "") or "nan"); lon = float(r.get("lon", "") or "nan")
            return math.isfinite(lat) and math.isfinite(lon)
        except Exception:
            return False
    miss = par_not.loc[~par_not.apply(_hasxy, axis=1)].copy()
    to_csv_like_source(miss, OUT_PAR_MISS, PARA_CSV)
    print(f"[OK] Wrote {OUT_PAR_MISS} ({len(miss)} rows)")

    # 5) MATCHES_summary.csv
    rows = []
    for i, j in exact_hits:
        rows.append({"PARATEC": paratec_df.at[i, "Nombre"], "OSM": osm_df.at[j, "name"], "type": "exact", "score": ""})
    for i, j, s in fuzzy_hits:
        rows.append({"PARATEC": paratec_df.at[i, "Nombre"], "OSM": osm_df.at[j, "name"], "type": "fuzzy", "score": f"{s:.0f}"})
    df_sum = pd.DataFrame(rows, columns=["PARATEC", "OSM", "type", "score"])
    to_csv_like_source(df_sum, OUT_MATCH_SUM, PARA_CSV)
    print(f"[OK] Wrote {OUT_MATCH_SUM} ({len(df_sum)} rows)")

    # 6) OSM_PARATEC_enriched.csv (solo los OSM que hicieron match)
    matched_j = {j for _, j in exact_hits} | {j for _, j, _ in fuzzy_hits}
    osm_min = osm_df.loc[osm_df.index.isin(matched_j), ["name", "lat", "lon"]].copy()
    to_csv_like_source(osm_min, OUT_OSM_ENR_MIN, OSM_CSV)
    print(f"[OK] Wrote {OUT_OSM_ENR_MIN} ({len(osm_min)} rows)")

    # 7) MATCHES_by_type.csv
    n_exact, n_fuzzy = len(exact_hits), len(fuzzy_hits)
    df_typ = pd.DataFrame(
        [{"type": "exact", "count": n_exact},
         {"type": "fuzzy", "count": n_fuzzy},
         {"type": "total", "count": n_exact + n_fuzzy}]
    )
    to_csv_like_source(df_typ, OUT_MATCH_TYPE, PARA_CSV)
    cov = 100.0 * (n_exact + n_fuzzy) / max(1, len(paratec_df))
    print(f"[STATS] exact={n_exact} fuzzy={n_fuzzy} total={n_exact+n_fuzzy} coverage={cov:.1f}%")
    print(f"[OK] Wrote {OUT_MATCH_TYPE} (coverage {cov:.1f}%)")

    return par_not  # para el summary


# ---------------- Main ----------------
def main():
    print("[LOAD] Reading inputs...")
    df_par_raw, par_meta = load_paratec()
    df_osm,     osm_meta = load_osm()

    # claves normalizadas (para uniques y summary)
    df_par_raw["_key"] = df_par_raw["Nombre"].map(norm_name_key)
    df_osm["_key"]     = df_osm["name"].map(norm_name_key)

    df_par      = df_par_raw.drop_duplicates(subset=["_key"]).reset_index(drop=True)
    df_osm_best = pick_best_by_key(df_osm, "_key")

    print("[MATCH] Running exact + fuzzy...")
    exact_hits, fuzzy_hits = run_matching(df_par, df_osm)

    print("[WRITE] Generating outputs...")
    par_not = enrich_and_write_outputs(df_par, df_osm, exact_hits, fuzzy_hits, par_meta, osm_meta)

    # conjuntos para console summary
    matched_osm_keys = set()
    for _, j in exact_hits:
        k = df_osm.at[j, "_key"]
        if k: matched_osm_keys.add(k)
    for _, j, _ in fuzzy_hits:
        k = df_osm.at[j, "_key"]
        if k: matched_osm_keys.add(k)

    all_osm_keys = set(df_osm_best["_key"].fillna("")); all_osm_keys.discard("")
    osm_not_keys = all_osm_keys - matched_osm_keys
    osm_not_rows = df_osm_best[df_osm_best["_key"].isin(osm_not_keys)].copy()

    to_csv_like_source(osm_not_rows[["name","lat","lon"]], OUT_OSM_NOT, OSM_CSV)
    print(f"[OK] Wrote {OUT_OSM_NOT} ({len(osm_not_rows)} rows)")

    feats2 = []
    for _, r in osm_not_rows.iterrows():
        try:
            lat = float(r.get("lat","") or "nan"); lon = float(r.get("lon","") or "nan")
        except Exception:
            lat, lon = float("nan"), float("nan")
        if not (math.isfinite(lat) and math.isfinite(lon)): continue
        feats2.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": r.get("name",""), "source": "OSM"},
        })
    with open(OUT_OSM_NOT_GJ, "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":feats2}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {OUT_OSM_NOT_GJ} ({len(feats2)} points)")

    # -------- Console summary--------
    print("--- Summary ---")
    print(f"PARATEC rows (raw):                {len(df_par_raw)}")
    print(f"PARATEC unique by Nombre:          {len(df_par)}")
    print(f"OSM rows (filtered):               {len(df_osm)}")
    print(f"OSM unique by key:                 {len(df_osm_best)}")
    print(f"Matched (total):                   {len(matched_osm_keys)}")
    print(f"Not in OSM (total):                {len(par_not)}")

    pct_par = 100.0 * len(matched_osm_keys) / len(df_par) if len(df_par) > 0 else 0.0
    print(f"Matched with XM (PARATEC):         {len(matched_osm_keys)} / {len(df_par)} ({pct_par:.1f}%)")

    # how many OSM uniques are not covered by XM
    print(f"OSM not in XM (total):             {len(osm_not_rows)}")
    pct_osm_covered = 100.0 * (len(df_osm_best) - len(osm_not_rows)) / len(df_osm_best) if len(df_osm_best) > 0 else 0.0
    print(f"OSM covered by XM:                 {len(df_osm_best) - len(osm_not_rows)} / {len(df_osm_best)} ({pct_osm_covered:.1f}%)")

    # resumen adicional (igual estilo)
    n_exact, n_fuzzy = len(exact_hits), len(fuzzy_hits)
    print(f"[DONE] Exact: {n_exact} | Fuzzy: {n_fuzzy} | Total: {n_exact+n_fuzzy} | Coverage: {100.0*(n_exact+n_fuzzy)/max(1,len(df_par)):.1f}%")

if __name__ == "__main__":
    main()
