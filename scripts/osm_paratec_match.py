#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, re, unicodedata, math, json
from collections import defaultdict
import pandas as pd
import numpy as np

# ---------------- Config ----------------
PARA_CSV = "PARATEC_with_coords.csv"
OSM_CSV  = "osm_substations_filtered.csv"

FUZZY_THRESHOLD = 70

# Outputs (los que ya estaban bien se mantienen)
OUT_PAR_ENR   = "PARATEC_enriched_coords.csv"
OUT_PAR_GJ    = "PARATEC_not_in_OSM.geojson"
OUT_PAR_MISS  = "PARATEC_not_in_OSM_missing_coords.csv"   # se sigue generando (no lo pediste eliminar)
OUT_MATCH_SUM = "MATCHES_summary.csv"                     # se mantiene

# NUEVOS (reemplazan 11–14)
OUT_OSM_ENR_MIN = "OSM_PARATEC_enriched.csv"              # (11)
OUT_PAR_NOT_CSV = "PARATEC_not_in_OSM.csv"                # (12, CSV compañero del GeoJSON)
OUT_MATCH_TYPE  = "MATCHES_by_type.csv"                   # (13, resumen mínimo)

# ------------- CSV I/O utils -------------

def sniff_encoding_delim_eol(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        enc = "latin-1"
    try:
        delim = csv.Sniffer().sniff("\n".join(text.splitlines()[:200]),
                                    delimiters=[",",";","\t","|"]).delimiter
    except Exception:
        delim = ","
    if "\r\n" in text[:1000]:
        eol = "\r\n"
    elif "\r" in text[:1000]:
        eol = "\r"
    else:
        eol = "\n"
    return enc, delim, eol

def read_csv_smart(path: str):
    enc, delim, eol = sniff_encoding_delim_eol(path)
    df = pd.read_csv(path, sep=delim, dtype=str, keep_default_na=False, na_values=[""],
                     encoding=enc, engine="python")
    df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df, enc, delim, eol

def fmt_lon_lat(v, decimals=7):
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return ""

def _sanitize_df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({np.nan: ""})
    for c in out.columns:
        out[c] = out[c].astype(str).str.replace(r"[\r\n]+", " ", regex=True)
    return out

def to_csv_like_source(df: pd.DataFrame, path: str, delim: str, enc: str, eol: str, columns=None):
    out = df.copy()
    for c in ["lon","lat","OSM_lon","OSM_lat","PAR_lon","PAR_lat"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_lon_lat)
    if "score" in out.columns:
        out["score"] = out["score"].apply(lambda v: "" if v=="" or pd.isna(v) else f"{float(v):.1f}")
    if columns:
        for c in columns:
            if c not in out.columns:
                out[c] = ""
        out = out[columns]
    out = _sanitize_df_for_csv(out)
    out.to_csv(path, index=False, sep=delim,
               encoding=("utf-8-sig" if enc.lower().startswith("utf") else enc),
               lineterminator=eol, quoting=csv.QUOTE_MINIMAL)

# ------------- Normalización / tokenización -------------

_ROMAN_MAP = {"i":"1","ii":"2","iii":"3","iv":"4","v":"5","vi":"6","vii":"7","viii":"8","ix":"9","x":"10"}
_STOPWORDS = {"subestacion","subestación","se","s/e","estacion","estación","san","santo","santa","sta","sto","sa",
              "calle","cll","av","avenida","norte","sur","este","oeste","oriente","occidente","de","del","la","el",
              "eeb","eeeb","bogota","bogotá"}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def normalize_core(name: str) -> str:
    if not isinstance(name, str): return ""
    s = strip_accents(name).lower().strip()
    s = re.sub(r"\(.*?\)", "", s) # removes aliases in parentheses
    s = re.sub(r"\b\d+(\.\d+)?\s*kv\b", "", s) #removes "115 kV", "34.5kv", etc.
    s = re.sub(r"\s+", " ", s).strip()
    return s

def roman_to_arabic_token(tok: str) -> str: return _ROMAN_MAP.get(tok, tok)
def arabic_to_roman_token(tok: str) -> str: return {v:k for k,v in _ROMAN_MAP.items()}.get(tok, tok)

def tokenize(name: str):
    s = re.sub(r"[^a-z0-9\s]+", " ", name)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        t1 = roman_to_arabic_token(t); out.append(t1)
        t2 = arabic_to_roman_token(t1)
        if t2 != t1: out.append(t2)
    return [t for t in out if t not in _STOPWORDS and (len(t)>1 or t.isdigit())]

def normalized_key(name: str) -> str:
    core = normalize_core(name)
    core = re.sub(r"[^a-z0-9]+", "", core)
    for r,a in _ROMAN_MAP.items():
        core = re.sub(rf"{r}(?![a-z0-9])", a, core)
    return core

# ------------- Matching helpers -------------

def build_blocks(keys_tokens: dict):
    by_initial = defaultdict(set); by_lenband = defaultdict(set); token_index = defaultdict(set)
    for k, toks in keys_tokens.items():
        if not k: continue
        by_initial[k[0]].add(k); by_lenband[len(k)//3].add(k)
        for t in set(toks): token_index[t].add(k)
    return by_initial, by_lenband, token_index

def candidate_set(k, toks, by_initial, by_lenband, token_index):
    cands = set()
    if k: cands |= by_initial.get(k[0], set()) | by_lenband.get(len(k)//3, set())
    for t in set(toks): cands |= token_index.get(t, set())
    return list(cands)

def score_pair(a_key, a_tokens, b_key, b_tokens, rf=None):
    sa, sb = set(a_tokens), set(b_tokens)
    jacc = 100.0 * (len(sa & sb) / max(1, len(sa | sb)))
    if rf is not None:
        from rapidfuzz import fuzz
        tset  = fuzz.token_set_ratio(a_key, b_key)
        tsort = fuzz.token_sort_ratio(a_key, b_key)
        part  = fuzz.partial_ratio(a_key, b_key)
        try:
            from rapidfuzz.distance import JaroWinkler
            jw = 100.0 * JaroWinkler.normalized_similarity(a_key, b_key)
        except Exception:
            jw = 0.0
    else:
        import difflib
        ratio = 100.0 * difflib.SequenceMatcher(None, a_key, b_key).ratio()
        tset = tsort = part = ratio; jw = 0.0
    return 0.35*tset + 0.25*tsort + 0.15*part + 0.15*jacc + 0.10*jw

def best_match_for(key, toks, osm_keys_tokens, blocks, rf=None):
    by_initial, by_lenband, token_index = blocks
    cands = candidate_set(key, toks, by_initial, by_lenband, token_index)
    if not cands: return None, 0.0
    best_k, best_s = None, -1.0
    for ck in cands:
        s = score_pair(key, toks, ck, osm_keys_tokens[ck], rf)
        if s > best_s: best_s, best_k = s, ck
    return best_k, best_s

def to_float(x):
    if x is None or x == "": return np.nan
    s = str(x).strip().replace(",", ".")
    try: return float(s)
    except ValueError: return np.nan

def df_to_geojson_points(df_subset: pd.DataFrame, out_path: str, lon_col="lon", lat_col="lat"):
    gj = {"type":"FeatureCollection","features":[]}
    for _, row in df_subset.iterrows():
        try: lon = float(row[lon_col]); lat = float(row[lat_col])
        except Exception: continue
        props = row.drop([lon_col, lat_col]).to_dict()
        gj["features"].append({"type":"Feature","properties":props,
                               "geometry":{"type":"Point","coordinates":[lon, lat]}})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

# ------------- Main -------------

def main():
    # Read inputs
    df_par_raw, par_enc, par_delim, par_eol = read_csv_smart(PARA_CSV)
    df_osm, osm_enc, osm_delim, osm_eol     = read_csv_smart(OSM_CSV)

    assert "Nombre" in df_par_raw.columns, "PARATEC: missing 'Nombre'"
    for c in ("lon","lat"):
        if c not in df_par_raw.columns: df_par_raw[c] = ""

    assert "name" in df_osm.columns, "OSM: missing 'name'"
    for c in ("lon","lat"):
        if c not in df_osm.columns: df_osm[c] = ""

    # Deduplicate PARATEC by exact Nombre
    df_par = df_par_raw.drop_duplicates(subset=["Nombre"], keep="first").copy()

    # Keys/tokens
    df_par["_key"]    = df_par["Nombre"].astype(str).map(normalized_key)
    df_par["_tokens"] = df_par["Nombre"].astype(str).map(normalize_core).map(tokenize)
    df_osm["_key"]    = df_osm["name"].astype(str).map(normalized_key)
    df_osm["_tokens"] = df_osm["name"].astype(str).map(normalize_core).map(tokenize)

    # Numeric coords
    for c in ("lon","lat"):
        df_par[c] = df_par[c].map(to_float)
        df_osm[c] = df_osm[c].map(to_float)

    # Unique OSM by key
    df_osm_best = df_osm.drop_duplicates("_key", keep="first").copy()

    # Exact + fuzzy mapping
    keys_par, keys_osm = set(df_par["_key"]), set(df_osm_best["_key"])
    exact_found   = keys_par & keys_osm
    exact_missing = keys_par - keys_osm

    osm_keys_tokens = {k: toks for k, toks in df_osm_best[["_key","_tokens"]].itertuples(index=False)}
    blocks = build_blocks(osm_keys_tokens)
    try:
        import rapidfuzz as rf  # noqa: F401
        rf_mod = rf
    except Exception:
        rf_mod = None

    df_par_missing = df_par[df_par["_key"].isin(exact_missing)].copy()
    best_keys, best_scores = [], []
    for k, toks in df_par_missing[["_key","_tokens"]].itertuples(index=False):
        bk, sc = best_match_for(k, toks, osm_keys_tokens, blocks, rf_mod)
        best_keys.append(bk); best_scores.append(sc)
    df_par_missing["best_osm_key"] = best_keys
    df_par_missing["best_score"]   = best_scores
    accepted = df_par_missing["best_score"] >= FUZZY_THRESHOLD

    # par_key -> (osm_key, match_type, score)
    par_to_osm = {k: (k, "exact", 100.0) for k in exact_found}
    for k, bk, sc in df_par_missing.loc[accepted, ["_key","best_osm_key","best_score"]].itertuples(index=False):
        par_to_osm[k] = (bk, "fuzzy", float(sc))

    # osm_key -> best par_key
    osm_to_par = {}
    for pk, (ok, mtype, sc) in par_to_osm.items():
        if ok not in osm_to_par or sc > osm_to_par[ok][2]:
            osm_to_par[ok] = (pk, mtype, sc)

    # ---------- PARATEC_enriched_coords (se mantiene igual) ----------
    osm_meta = df_osm_best.set_index("_key")[["lon","lat"]].to_dict(orient="index")
    df_par_enr = df_par.copy()
    def fill_coords(row):
        if not pd.isna(row["lon"]) and not pd.isna(row["lat"]):
            return row["lon"], row["lat"]
        k = row["_key"]; mapping = par_to_osm.get(k)
        if mapping:
            ok = mapping[0]; m = osm_meta.get(ok, {})
            if pd.notna(m.get("lon")) and pd.notna(m.get("lat")):
                return m["lon"], m["lat"]
        return row["lon"], row["lat"]
    filled = df_par_enr.apply(lambda r: fill_coords(r), axis=1, result_type="reduce")
    df_par_enr["lon"] = [x[0] for x in filled]; df_par_enr["lat"] = [x[1] for x in filled]
    to_csv_like_source(df_par_enr[list(df_par_raw.columns)], OUT_PAR_ENR, par_delim, par_enc, par_eol)
    print(f"Wrote {OUT_PAR_ENR}")

    # ---------- Not in OSM (GeoJSON existente + NUEVO CSV compacto) ----------
    par_not = df_par[~df_par["_key"].isin(par_to_osm.keys())].copy()
    par_not_with = par_not[pd.notna(par_not["lon"]) & pd.notna(par_not["lat"])].copy()
    par_not_without = par_not[~(pd.notna(par_not["lon"]) & pd.notna(par_not["lat"]))].copy()
    # GeoJSON (igual que antes)
    df_to_geojson_points(par_not_with, OUT_PAR_GJ, lon_col="lon", lat_col="lat")
    # CSV nuevo con TODO PARATEC (coords si existen)
    to_csv_like_source(par_not[list(df_par_raw.columns)], OUT_PAR_NOT_CSV, par_delim, par_enc, par_eol)
    print(f"Wrote {OUT_PAR_GJ} and {OUT_PAR_NOT_CSV}")
    # (Seguimos escribiendo la lista de faltantes sin coords por si te sirve)
    to_csv_like_source(par_not_without[list(df_par_raw.columns)], OUT_PAR_MISS, par_delim, par_enc, par_eol)
    print(f"Wrote {OUT_PAR_MISS}")

    # ---------- (11) OSM_PARATEC_enriched: SOLO matched, OSM coords+name + TODAS columnas PARATEC (sin lon/lat) ----------
    par_cols_no_coords = [c for c in df_par_raw.columns if c not in ("lon","lat")]
    matched_osm_keys = list(osm_to_par.keys())
    base = df_osm_best[df_osm_best["_key"].isin(matched_osm_keys)].copy()

    # Trae la clave PAR y fusiona atributos PARATEC (sin lon/lat)
    base["PAR_key"] = base["_key"].map(lambda ok: (osm_to_par.get(ok) or (None,None,None))[0])
    par_attrs = df_par_enr[["_key"] + par_cols_no_coords].rename(columns={"_key":"PAR_key"})
    df_osm_enriched = base.merge(par_attrs, on="PAR_key", how="left")

    # Columnas finales: OSM lon/lat/name + columnas PARATEC (orden original, sin lon/lat)
    final_cols_11 = ["lon","lat","name"] + par_cols_no_coords
    df_osm_enriched = df_osm_enriched[final_cols_11].copy()
    to_csv_like_source(df_osm_enriched, OUT_OSM_ENR_MIN, osm_delim, osm_enc, osm_eol)
    print(f"Wrote {OUT_OSM_ENR_MIN}")

    # ---------- (13) MATCHES_by_type: solo nombres y score ----------
    rows = []
    for pk, (ok, mtype, sc) in par_to_osm.items():
        p_name = df_par.loc[df_par["_key"]==pk, "Nombre"].iloc[0] if (df_par["_key"]==pk).any() else ""
        o_name = df_osm_best.loc[df_osm_best["_key"]==ok, "name"].iloc[0] if (df_osm_best["_key"]==ok).any() else ""
        rows.append({
            "PARATEC_Nombre": p_name,
            "OSM_name": o_name,
            "match_type": mtype,
            "score": f"{float(sc):.1f}"
        })
    df_matches_min = pd.DataFrame(rows, columns=["PARATEC_Nombre","OSM_name","match_type","score"])
    to_csv_like_source(df_matches_min, OUT_MATCH_TYPE, osm_delim, osm_enc, osm_eol)
    print(f"Wrote {OUT_MATCH_TYPE}")

    # (Se mantiene el MATCHES_summary existente por compatibilidad)
    to_csv_like_source(df_matches_min.rename(columns={
        "PARATEC_Nombre":"PAR_Nombre","OSM_name":"OSM_name"
    }), OUT_MATCH_SUM, osm_delim, osm_enc, osm_eol)
    print(f"Wrote {OUT_MATCH_SUM}")

    # Console summary
    print("--- Summary ---")
    print(f"PARATEC rows (raw):                {len(df_par_raw)}")
    print(f"PARATEC unique by Nombre:          {len(df_par)}")
    print(f"OSM rows (filtered):               {len(df_osm)}")
    print(f"OSM unique by key:                 {len(df_osm_best)}")
    print(f"Matched (total):                   {len(matched_osm_keys)}")
    print(f"Not in OSM (total):                {len(par_not)}")

    pct = 100.0 * len(matched_osm_keys) / len(df_par) if len(df_par) > 0 else 0.0
    print(f"OSM Substation matched with XM-UPME dataset:         {len(matched_osm_keys)} / {len(df_par)} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
