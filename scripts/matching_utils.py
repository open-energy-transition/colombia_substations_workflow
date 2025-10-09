# -*- coding: utf-8 -*-
"""
matching_utils.py

Single source of truth for ALL matching functionality:
- Voltage stripping and heavy station name cleanup (parentheses, admin tails)
- Roman↔Arabic numerals, tokenization, stopwords, domain aliases
- Canonical station key builders (strict, relaxed, station, station_plus)
- Fuzzy helpers (RapidFuzz if present; safe fallback)
- Coordinate helpers: Haversine + optional BallTree (scikit-learn) with meters output

Only this file should contain matching logic. Main scripts should import and use these helpers.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional
import re
import unicodedata
import math

# ------------------------ Optional deps ------------------------
HAS_RAPIDFUZZ = False
RF_FUZZ = None
RF_JW = None  

try:
    from rapidfuzz import fuzz as RF_FUZZ  # token_set_ratio, token_sort_ratio, etc.
    try:
        
        from rapidfuzz.distance import JaroWinkler as _RF_JW_CLASS  # type: ignore
        RF_JW = _RF_JW_CLASS
    except Exception:
        
        RF_JW = None
    HAS_RAPIDFUZZ = True
except Exception:
    RF_FUZZ = None
    RF_JW = None
    HAS_RAPIDFUZZ = False

# scikit-learn BallTree (optional)
try:
    from sklearn.neighbors import BallTree as _BallTree 
except Exception:
    _BallTree = None

# ------------------------ Tunables -----------------------------
FUZZY_THRESHOLD = 70     # You can override from the main script
DEFAULT_RADIUS_M = 3000.0

# ------------------------ Basic cleaning -----------------------
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9a-zA-ZñÑáéíóúÁÉÍÓÚäëïöüÄËÏÖÜ\s]+")

def _collapse_ws(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()

def strip_accents(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def normalize_core(s: str) -> str:
    s = strip_accents(s).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _collapse_ws(s)

# ------------------------ Voltage & noise ----------------------
# e.g. "115 kV", "34.5kv", "230-kV", "500 / kV"
_VOLTAGE_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*[-/]?\s*k\s*?v\b", re.IGNORECASE)
def strip_voltage_tokens(s: str) -> str:
    if s is None:
        return ""
    return _collapse_ws(_VOLTAGE_RE.sub(" ", str(s)))

# parentheses (e.g., "(NARIÑO)")
_PARENS_RE = re.compile(r"\([^)]*\)")
# trailing admin tail after dash/comma (e.g., " - Antioquia", ", Bogotá")
_TRAIL_ADMIN_RE = re.compile(r"[\s]*[-–—,]\s*[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]+$")
# squeeze multiple separators
_MULTI_SEP_RE = re.compile(r"[/_\-–—]{2,}")

def heavy_station_clean(s: str) -> str:
    """Heavier cleanup for station names: voltage, parentheses, trailing '- Departamento', repeated separators."""
    s = strip_voltage_tokens(s)
    s = _PARENS_RE.sub(" ", s)
    s = _TRAIL_ADMIN_RE.sub(" ", s)
    s = _MULTI_SEP_RE.sub(" ", s)
    s = s.replace("-", " ").replace("–", " ").replace("—", " ").replace("/", " ")
    return normalize_core(s)

# ------------------------ Romans & tokens ----------------------
_ROMAN_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15
}
def roman_to_arabic_token(tok: str) -> str:
    t = tok.lower()
    return str(_ROMAN_MAP[t]) if t in _ROMAN_MAP else tok

def arabic_to_roman_token(tok: str) -> str:
    # Minimal reverse mapping for small numerals used in station names
    try:
        n = int(tok)
    except Exception:
        return tok
    rev = {v: k.upper() for k, v in _ROMAN_MAP.items()}
    return rev.get(n, tok)

_STOPWORDS = {
    "subestacion", "subestación", "sub", "se", "estacion", "estación",
    "de", "del", "la", "el", "los", "las", "y", "the",
}

_ALIAS_MAP = {
    "sta": "santa", "sta.": "santa", "sto": "san", "sto.": "san",
    "sn": "san", "snta": "santa", "nte": "n", "norte": "n",
    "sur": "s", "occidente": "o", "oriente": "e", "este": "e", "oeste": "o",
}

def _alias(tok: str) -> str:
    return _ALIAS_MAP.get(tok, tok)

def tokenize(s: str) -> List[str]:
    s = normalize_core(s)
    toks = [t for t in s.split(" ") if t]
    out = []
    for t in toks:
        t = roman_to_arabic_token(t)
        t = _alias(t)
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out

def collapse_repeats(key: str) -> str:
    toks = key.split()
    out = []
    seen = set()
    for t in toks:
        if (t, len(out)) not in seen:
            out.append(t)
            seen.add((t, len(out)))
    return " ".join(out)

# ------------------------ Key builders ------------------------
def normalized_key_strict(name: str) -> str:
    """Aggressive generic normalization, no domain extras."""
    toks = tokenize(name)
    return collapse_repeats(" ".join(toks))

def normalized_key_relaxed(name: str) -> str:
    """Light cleanup; keep more tokens (useful before fuzzy)."""
    return normalize_core(name)

def normalized_key_station(name: str) -> str:
    """Domain-aware, but without voltage-specific extras."""
    base = normalize_core(name)
    toks = tokenize(base)
    return collapse_repeats(" ".join(toks))

def normalized_key_station_plus(name: str) -> str:
    """Strongest: heavy cleanup (strips voltage, parens, admin tails) + domain aliases + roman→arabic."""
    base = heavy_station_clean(name)
    toks = tokenize(base)
    return collapse_repeats(" ".join(toks))

# ------------------------ Fuzzy helpers ------------------------
def fuzzy_score(a: str, b: str) -> float:
    """Composite fuzzy score on normalized inputs (0..100)."""
    a_key = normalized_key_relaxed(a)
    b_key = normalized_key_relaxed(b)
    if not a_key or not b_key:
        return 0.0

    # token Jaccard (cheap + robust)
    a_tokens = set(a_key.split())
    b_tokens = set(b_key.split())
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jacc = (100.0 * inter / union) if union else 0.0

    # space-insensitive strings
    a_ns = a_key.replace(" ", "")
    b_ns = b_key.replace(" ", "")

    if HAS_RAPIDFUZZ and RF_FUZZ is not None:
        tset  = RF_FUZZ.token_set_ratio(a_key, b_key)
        tsort = RF_FUZZ.token_sort_ratio(a_key, b_key)
        part  = RF_FUZZ.partial_ratio(a_key, b_key)
        jw    = 100.0 * RF_JW.normalized_similarity(a_key, b_key) if RF_JW else 0.0
        # character-level, no-spaces ratio
        ns_ch = RF_FUZZ.ratio(a_ns, b_ns)
    else:
        import difflib
        a_tok = " ".join(sorted(a_tokens))
        b_tok = " ".join(sorted(b_tokens))
        ratio = 100.0 * difflib.SequenceMatcher(None, a_tok, b_tok).ratio()
        tset = tsort = part = ratio
        jw   = 0.0
        ns_ch = 100.0 * difflib.SequenceMatcher(None, a_ns, b_ns).ratio()

    # Blend: bump weight on no-space char match slightly to favor compounds
    return 0.30*tset + 0.22*tsort + 0.13*part + 0.15*jacc + 0.10*jw + 0.10*ns_ch

# ------------------------ Coordinates -------------------------
EARTH_R_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Return great-circle distance in meters."""
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(float(lat1))) *
         math.cos(math.radians(float(lat2))) *
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R_M * c

def _deg_to_rad(series):
    import numpy as np
    return np.radians(series.astype(float).values)

def _to_radians(lat_series, lon_series):
    import numpy as np

    EARTH_R_M = 6371000.0

    def _deg_to_rad(series):
        return np.radians(series.astype(float).values)

    def _to_radians(lat_series, lon_series):
        """Return Nx2 array in radians for BallTree(haversine)."""
        lat = _deg_to_rad(lat_series)
        lon = _deg_to_rad(lon_series)
        return np.vstack([lat, lon]).T

def build_balltree_haversine(lat_series, lon_series):
    """Build a BallTree (haversine metric). Raises if sklearn is missing."""
    if _BallTree is None:
        raise RuntimeError("scikit-learn is not available; cannot build BallTree.")
    pts = _to_radians(lat_series, lon_series)
    tree = _BallTree(pts, metric="haversine")
    return tree, pts

def tree_knearest_m(tree, query_rad, k=1):
    """Return (dist_m, idx) for nearest neighbors on a haversine BallTree."""
    dist_rad, idx = tree.query(query_rad, k=k)
    dist_m = dist_rad * EARTH_R_M
    return dist_m, idx
