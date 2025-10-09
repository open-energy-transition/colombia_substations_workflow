# -*- coding: utf-8 -*-
"""
enrich_xm_with_coords.py

Goal:
- Enrich PARATEC_substations.csv with lon/lat coordinates from XM_enriched_with_UPME_coords.csv
  using NAME-ONLY matching (PARATEC has no coordinates).
- Many-to-one allowed: multiple PARATEC rows (per voltage) can match the same XM station.
- Write the enriched PARATEC in the SAME DIALECT (sep/encoding/quote/newline) as the input PARATEC,
  and ensure tildes (áéíóúñ) display correctly in Excel.

Outputs -> ./outputs_paratec_name_only
  - PARATEC_enriched_with_XMcoords.csv        (same “style” as input PARATEC)
  - PARATEC_XM_matches.csv                    (row-level matches: method/score)
  - PARATEC_unmatched_diagnostics.csv         (PARATEC rows that found no XM match)
  - XM_unmatched_in_PARATEC.csv               (XM stations not present in PARATEC, by normalized key)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys, importlib.util
import csv
import numpy as np
import pandas as pd

# ---------------------- CONFIG ----------------------
DEFAULTS = {
    "xm_enriched_csv": Path("XM_enriched_with_UPME_coords.csv"),
    "paratec_csv": Path("PARATEC_substations.csv"),
    # "outdir": Path("outputs_paratec_name_only"),

    # Matching
    "use_fuzzy": True,
    "fuzzy_threshold_override": 85,     # None => utils.FUZZY_THRESHOLD

    # Optional: explicit path to matching_utils.py
    "utils_path": None,                 # None => import from PYTHONPATH or same folder
}
# ---------------------------------------------------


# ------------------------------- CLI ----------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Enrich PARATEC CSV with XM coords (paths as args).")
    p.add_argument("--paratec", dest="paratec_csv", help="Path to PARATEC_substations.csv (from step 2).")
    p.add_argument("--xm", dest="xm_enriched_csv", help="Path to XM_enriched_with_UPME_coords.csv (from step 5).")
    p.add_argument("--utils-path", dest="utils_path", help="Explicit path to matching_utils.py (optional).")
    p.add_argument("--fuzzy-threshold", type=int, dest="fuzzy_threshold_override", help="Override fuzzy threshold.")
    p.add_argument("--out-enriched", dest="out_enriched", help='Output CSV for PARATEC_enriched_with_XMcoords.csv')
    p.add_argument("--out-matches", dest="out_matches", help='Output CSV for PARATEC_XM_matches.csv')
    p.add_argument("--out-unmatched-pr", dest="out_unmatched_pr", help='Output CSV for PARATEC_unmatched_diagnostics.csv')
    p.add_argument("--out-xm-unmatched", dest="out_xm_unmatched", help='Output CSV for XM_unmatched_in_PARATEC.csv')
    return p.parse_args()

# ---------------------- Robust CSV loading + dialect capture ----------------------
def _detect_newline(sample_bytes: bytes) -> str:
    if b"\r\n" in sample_bytes:
        return "\r\n"
    return "\n"

def _sniff_dialect(sample_text: str, default_sep: str = ",") -> Tuple[str, str]:
    """Return (sep, quotechar). If sniffer fails, default to (default_sep, '\"')."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",",";","\t","|"])
        sep = dialect.delimiter or default_sep
        qc  = dialect.quotechar or '"'
        return sep, qc
    except Exception:
        return default_sep, '"'

def read_csv_with_meta(path: str | Path) -> Tuple[pd.DataFrame, dict]:
    """
    Robust CSV reader that returns (DataFrame, meta):
      meta = {"sep": ..., "encoding": ..., "quotechar": ..., "newline": ...}
    - Sniffs delimiter/quotechar/newline from first 64KB
    - Tries encodings: utf-8, utf-8-sig, latin-1, cp1252
    - Uses engine='python', on_bad_lines='skip'
    - Reads everything as strings
    """
    path = str(path)
    with open(path, "rb") as f:
        sample = f.read(65536)
    newline = _detect_newline(sample)
    # decode sample to sniff dialect
    try:
        sample_text = sample.decode("utf-8", errors="ignore")
    except Exception:
        sample_text = sample.decode("latin-1", errors="ignore")
    sep, quotechar = _sniff_dialect(sample_text, default_sep=",")

    errors_seen = []
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                encoding=enc,
                dtype=str,
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
                quotechar=quotechar,
            )
            # strip BOM from headers if present
            df.columns = [c.replace("\ufeff", "") if isinstance(c, str) else c for c in df.columns]
            meta = {"sep": sep, "encoding": enc, "quotechar": quotechar, "newline": newline}
            print(f"[INFO] Loaded {Path(path).name} sep='{sep}' enc='{enc}' quotechar='{quotechar}' rows={len(df)} cols={len(df.columns)}")
            return df, meta
        except Exception as e:
            errors_seen.append((enc, str(e)[:180]))

    msg = "\n".join([f"- {enc}: {err}" for enc, err in errors_seen])
    raise RuntimeError(f"Failed to read CSV {path} with common encodings.\nTried:\n{msg}")


# ----------------- Import matching_utils -----------------
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


# ----------------- Small helpers -----------------
def pick_name_col(df: pd.DataFrame, candidates=("Nombre","name","NOMBRE")) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def choose_lon_lat_columns(xm: pd.DataFrame) -> Tuple[str, str]:
    """
    Prefer 'lon'/'lat' if present; else fall back to 'longitude'/'latitude'.
    Returns tuple (lon_col, lat_col) or raises if neither exists.
    """
    lon_col = None
    lat_col = None
    for cand in ("lon", "longitude"):
        if cand in xm.columns:
            lon_col = cand
            break
    for cand in ("lat", "latitude"):
        if cand in xm.columns:
            lat_col = cand
            break
    if lon_col is None or lat_col is None:
        raise ValueError("XM file must contain lon/lat or longitude/latitude columns.")
    return lon_col, lat_col

def format_coord_for_dialect(val: float | str | None, sep: str, places: int = 6) -> str:
    """
    Convert numeric lon/lat to string for CSV dialect:
    - if sep == ';' -> use comma decimals (Spanish Excel)
    - else -> dot decimals
    Returns '' for missing values.
    """
    if val is None:
        return ""
    try:
        f = float(val)
    except Exception:
        s = str(val).strip()
        return s if s.lower() != "nan" else ""
    s = f"{f:.{places}f}"
    if sep == ";":
        s = s.replace(".", ",")
    return s


# --------------------------------- MAIN ---------------------------------
def main():
    cfg = DEFAULTS.copy()
    args = parse_args()
    for k in ['xm_enriched_csv','paratec_csv','utils_path','fuzzy_threshold_override']:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    outdir = Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    mu = import_matching_utils(cfg["utils_path"])
    FUZZY_THRESHOLD = (cfg["fuzzy_threshold_override"]
                       if cfg["fuzzy_threshold_override"] is not None
                       else getattr(mu, "FUZZY_THRESHOLD", 90))

    # Load inputs WITH METADATA (so we can write back in same style as PARATEC)
    xm, _xm_meta = read_csv_with_meta(cfg["xm_enriched_csv"])
    pr, pr_meta  = read_csv_with_meta(cfg["paratec_csv"])
    pr_sep, pr_enc, pr_qc, pr_nl = pr_meta["sep"], pr_meta["encoding"], pr_meta["quotechar"], pr_meta["newline"]

    # Names
    if "elementName" not in xm.columns:
        raise ValueError("XM_enriched_with_UPME_coords.csv must contain column 'elementName'.")
    xm_name_col = "elementName"
    pr_name_col = pick_name_col(pr, ("Nombre","name","NOMBRE"))
    if not pr_name_col:
        raise ValueError("Could not find PARATEC name column (Nombre/name/NOMBRE).")

    # Ensure lon/lat columns exist in PARATEC (filled on match). Keep them as strings for exact style.
    if "lon" not in pr.columns:
        pr["lon"] = ""
    if "lat" not in pr.columns:
        pr["lat"] = ""

    # XM lon/lat columns (choose best available)
    xm_lon_col, xm_lat_col = choose_lon_lat_columns(xm)
    xm[xm_lon_col] = pd.to_numeric(xm[xm_lon_col], errors="coerce")
    xm[xm_lat_col] = pd.to_numeric(xm[xm_lat_col], errors="coerce")

    # Voltage-stripped display names (to avoid 115 kV etc.)
    pr["__name_par"] = pr[pr_name_col].astype(str).map(mu.strip_voltage_tokens).str.strip()
    xm["__name_xm"]  = xm[xm_name_col].astype(str).map(mu.strip_voltage_tokens).str.strip()

    # Canonical station keys (heavy normalization preferred)
    key_fn = (getattr(mu, "normalized_key_station_plus", None)
              or getattr(mu, "normalized_key_station", None)
              or getattr(mu, "normalized_key_strict"))

    pr["__key"] = pr["__name_par"].map(lambda s: key_fn(str(s)))
    xm["__key"] = xm["__name_xm"].map(lambda s: key_fn(str(s)))

    # Index from XM keys -> candidate XM rows (normally 1 per key in your file)
    xm_by_key: Dict[str, pd.DataFrame] = {k: g for k, g in xm.groupby("__key", dropna=True)}

    # ------------------ Match PARATEC -> XM (name-only) ------------------
    matches: List[Dict] = []

    for i, prr in pr.iterrows():
        chosen = None

        # 1) exact by normalized key
        k = prr["__key"]
        if isinstance(k, str) and k:
            cand = xm_by_key.get(k)
            if cand is not None and not cand.empty:
                # Prefer rows that have coordinates (should be all)
                cand_geo = cand.dropna(subset=[xm_lon_col, xm_lat_col])
                xm_row = cand_geo.iloc[0] if not cand_geo.empty else cand.iloc[0]
                chosen = ("name_exact", int(xm_row.name))

        # 2) fuzzy (display names)
        if chosen is None and cfg["use_fuzzy"]:
            pname = str(prr["__name_par"])
            best_j, best_sc = None, -1.0
            for j, xr in xm.iterrows():
                sc = mu.fuzzy_score(pname, str(xr["__name_xm"]))
                if sc > best_sc:
                    best_sc, best_j = sc, j
            if best_j is not None and best_sc >= FUZZY_THRESHOLD:
                chosen = ("name_fuzzy", int(best_j), float(best_sc))

        # record decision
        if chosen is not None:
            if chosen[0] == "name_exact":
                method, xm_idx = chosen
                aux = np.nan
            else:
                method, xm_idx, aux = chosen

            matches.append({
                "paratec_index": i,
                "paratec_name": prr["__name_par"],
                "paratec_key": prr["__key"],
                "xm_index": xm_idx,
                "xm_name": xm.at[xm_idx, "__name_xm"],
                "xm_key": xm.at[xm_idx, "__key"],
                "method": method,
                "score": aux if method == "name_fuzzy" else np.nan,
            })

    df_matches = pd.DataFrame(matches)

    # ------------------ Enrich PARATEC with XM lon/lat ONLY (respecting PARATEC dialect) ------------------
    enriched = pr.copy()
    replaced_count = 0

    if not df_matches.empty:
        for _, row in df_matches.iterrows():
            i = int(row["paratec_index"])
            j = int(row["xm_index"])
            x_lon = xm.at[j, xm_lon_col]
            x_lat = xm.at[j, xm_lat_col]
            lon_s = format_coord_for_dialect(x_lon, pr_sep, places=6)
            lat_s = format_coord_for_dialect(x_lat, pr_sep, places=6)
            if lon_s and lat_s:
                enriched.at[i, "lon"] = lon_s
                enriched.at[i, "lat"] = lat_s
                replaced_count += 1

    # ------------------ Unmatched PARATEC diagnostics (name-only) ------------------
    unmatched_idx = set(range(len(pr))) - set(df_matches["paratec_index"].tolist() if not df_matches.empty else [])
    diag_rows = []

    for i in sorted(unmatched_idx):
        rr = pr.loc[i]
        pname = str(rr["__name_par"]); pkey = rr["__key"]

        # best fuzzy candidate in XM
        best_j, best_sc, best_name = (None, -1.0, None)
        if DEFAULTS["use_fuzzy"]:
            for j, xr in xm.iterrows():
                sc = mu.fuzzy_score(pname, str(xr["__name_xm"]))
                if sc > best_sc:
                    best_sc, best_j = sc, j
            if best_j is not None:
                best_name = xm.at[best_j, "__name_xm"]

        key_in_xm = bool(isinstance(pkey, str) and pkey and (pkey in xm_by_key))
        if key_in_xm:
            reason = "same_name_exists_in_xm_but_not_selected"
        elif best_sc >= FUZZY_THRESHOLD:
            reason = "fuzzy_match_below_policy"
        else:
            reason = "no_name_match"

        diag_rows.append({
            "paratec_index": i,
            "paratec_name": pname,
            "paratec_key": pkey,
            "best_fuzzy_xm_index": best_j,
            "best_fuzzy_xm_name": best_name,
            "best_fuzzy_xm_score": best_sc if best_sc >= 0 else np.nan,
            "reason": reason,
        })

    df_unmatched_paratec = pd.DataFrame(diag_rows).sort_values(["reason","paratec_name"]).reset_index(drop=True)

    # ------------------ XM coverage check (by station key, not by row index) ------------------
    xm_keys = pd.Series(xm["__key"], dtype=str).dropna().unique().tolist()
    pr_keys = set(pd.Series(pr["__key"], dtype=str).dropna().unique().tolist())

    # Only report XM keys truly missing from PARATEC
    missing_keys = [k for k in xm_keys if k not in pr_keys]

    df_xm_unmatched_rows = []
    if missing_keys:
        xm_rep = (xm.dropna(subset=["__key"])
                    .groupby("__key", as_index=False)
                    .first()
                    .set_index("__key"))
        for k in missing_keys:
            if k not in xm_rep.index:
                continue
            xr = xm_rep.loc[k]
            xname = str(xr["__name_xm"])
            # best fuzzy candidate in PARATEC
            best_i, best_sc, best_name = (None, -1.0, None)
            if DEFAULTS["use_fuzzy"]:
                for i, prr in pr.iterrows():
                    sc = mu.fuzzy_score(xname, str(prr["__name_par"]))
                    if sc > best_sc:
                        best_sc, best_i = sc, i
                if best_i is not None:
                    best_name = pr.at[best_i, "__name_par"]

            df_xm_unmatched_rows.append({
                "xm_name": xname,
                "xm_key": k,
                "best_fuzzy_paratec_index": best_i,
                "best_fuzzy_paratec_name": best_name,
                "best_fuzzy_paratec_score": best_sc if best_sc >= 0 else np.nan,
                "reason": "xm_key_not_in_paratec",
            })

    df_xm_unmatched = pd.DataFrame(df_xm_unmatched_rows).sort_values(["xm_name"]).reset_index(drop=True)

    # ------------------ Save outputs ------------------
    outdir.mkdir(parents=True, exist_ok=True)
    out_enriched      = args.out_enriched or "PARATEC_enriched_with_XMcoords.csv"
    out_matches       = args.out_matches or "PARATEC_XM_matches.csv"
    out_unmatched_pr  = args.out_unmatched_pr or "PARATEC_unmatched_diagnostics.csv"
    out_xm_unmatched  = args.out_xm_unmatched or "XM_unmatched_in_PARATEC.csv"

    # Choose encoding to preserve tildes in Excel:
    save_enc = pr_enc
    if pr_enc.lower().startswith("utf-8"):
        save_enc = "utf-8-sig"

    # Write enriched with same *style* as input PARATEC
    enriched.to_csv(
        out_enriched,
        index=False,
        sep=pr_sep,
        encoding=save_enc,
        quoting=csv.QUOTE_MINIMAL,
        quotechar=pr_qc,
        lineterminator=pr_nl,
        na_rep="",  # blanks instead of 'NaN'
    )

    # Reports in UTF-8 with BOM (safe tildes in Excel)
    df_matches.to_csv(out_matches, index=False, encoding="utf-8-sig")
    df_unmatched_paratec.to_csv(out_unmatched_pr, index=False, encoding="utf-8-sig")
    df_xm_unmatched.to_csv(out_xm_unmatched, index=False, encoding="utf-8-sig")

    # ------------------ Console report ------------------
    print("\n--- PARATEC - XM (NAME-ONLY) Enrichment Report ---")
    print(f"PARATEC rows: {len(pr)}")
    print(f"XM enriched rows: {len(xm)}")
    print(f"Fuzzy: {'ON' if cfg['use_fuzzy'] else 'OFF'} (threshold={FUZZY_THRESHOLD})\n")

    print(f"Matched PARATEC rows: {len(df_matches)}")
    print(f"Coordinates written in PARATEC rows: {replaced_count}")
    print(f"Unmatched PARATEC rows (diagnostics): {len(df_unmatched_paratec)}")
    print(f"XM stations (by key) NOT present in PARATEC: {len(df_xm_unmatched)}\n")

    print("[OK] Saved files (enriched preserves original PARATEC style):")
    print(f"- {out_enriched}")
    print(f"- {out_matches}")
    print(f"- {out_unmatched_pr}")
    print(f"- {out_xm_unmatched}\n")


if __name__ == "__main__":
    main()
