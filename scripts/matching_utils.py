# -*- coding: utf-8 -*-
"""
matching_utils.py

Shared name-matching utilities for PARATEC/UPME/OSM workflows.

Exports (stable API):
- FUZZY_THRESHOLD
- HAS_RAPIDFUZZ
- _ROMAN_MAP
- _STOPWORDS
- strip_accents
- normalize_core
- roman_to_arabic_token
- arabic_to_roman_token
- tokenize
- normalized_key_strict
- normalized_key_relaxed
- normalized_key  (alias to strict for backward-compat)
- collapse_repeats
- build_blocks
- candidate_set
- score_pair
"""

from __future__ import annotations
import re
import unicodedata
from collections import defaultdict

# --- RapidFuzz (optional) at module import ---
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

# Default fuzzy acceptance threshold (0-100)
FUZZY_THRESHOLD: int = 63

# Roman numerals (lowercase) map used both directions
_ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
}

# Domain stopwords: generic facility terms, articles, directions, abbreviations,
# brand prefixes that often appear in OSM names, and frequent local tokens.
_STOPWORDS = {
    "subestacion","subestación","se","s/e","estacion","estación",
    "san","santo","santa","sta","sto","sa","calle","cll","av","avenida",
    "norte","sur","este","oeste","oriente","occidente","de","del","la","el",
    "eeb","eeeb","bogota","bogotá",
    "celsia",  # brand prefix often present in OSM facility names
}

# -----------------------------
# Normalization / tokenization
# -----------------------------
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )

def normalize_core(name: str, keep_parens: bool = False) -> str:
    """
    Lowercase, remove accents, remove voltage suffixes, collapse whitespace.
    If keep_parens=True, treat (), -, _, / as separators (keep content as tokens);
    otherwise remove parenthesized segments entirely.
    """
    if not isinstance(name, str):
        return ""
    s = strip_accents(name).lower().strip()
    if keep_parens:
        s = re.sub(r"[()\-\_/]+", " ", s)
    else:
        s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\b\d+(\.\d+)?\s*kv\b", "", s)    # remove "115 kV", "34.5kv", etc.
    s = re.sub(r"\s+", " ", s).strip()
    return s

def roman_to_arabic_token(tok: str) -> str:
    return _ROMAN_MAP.get(tok, tok)

def arabic_to_roman_token(tok: str) -> str:
    inv = {v: k for k, v in _ROMAN_MAP.items()}
    return inv.get(tok, tok)

def tokenize(name: str):
    """
    Tokenize a normalized string into domain-meaningful tokens, augmenting with
    roman/arabic variants and removing stopwords.
    """
    s = re.sub(r"[^a-z0-9\s]+", " ", name)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        t1 = roman_to_arabic_token(t); out.append(t1)
        t2 = arabic_to_roman_token(t1)
        if t2 != t1:
            out.append(t2)
    out = [t for t in out if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]
    return out

def _collapse_non_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s)

def normalized_key_strict(name: str) -> str:
    """
    Strict key: preserves all tokens (does NOT remove stopwords).
    Matches the original behavior you had in the pipeline.
    """
    core = normalize_core(name)
    core = _collapse_non_alnum(core)
    for r, a in _ROMAN_MAP.items():
        core = re.sub(rf"{r}(?![a-z0-9])", a, core)
    return core

def normalized_key_relaxed(name: str) -> str:
    """
    Relaxed key: removes domain stopwords / brand prefixes and collapses the rest.
    Useful for variants.
    """
    core = normalize_core(name)
    words = [w for w in core.split() if w not in _STOPWORDS]
    core2 = "".join(words)
    for r, a in _ROMAN_MAP.items():
        core2 = re.sub(rf"{r}(?![a-z0-9])", a, core2)
    core2 = _collapse_non_alnum(core2)
    return core2

# Backward-compat alias (your scripts call normalized_key)
normalized_key = normalized_key_strict

def collapse_repeats(s: str) -> str:
    """Collapse repeated letters. Keeps digits unchanged."""
    return re.sub(r"([a-z])\1+", r"\1", s)

# -----------------------------
# Candidate blocking & scoring
# -----------------------------
def build_blocks(keys_tokens: dict):
    by_initial = defaultdict(set)
    by_lenband = defaultdict(set)
    token_index = defaultdict(set)
    for k, toks in keys_tokens.items():
        if not k:
            continue
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
    """
    Composite 0–100 score combining token-based similarities.
    Uses RapidFuzz when available; otherwise falls back to difflib on token-sorted strings.
    """
    sa, sb = set(a_tokens), set(b_tokens)
    denom = len(sa | sb)
    jacc = 100.0 * (len(sa & sb) / denom) if denom else 0.0

    if HAS_RAPIDFUZZ and fuzz is not None:
        tset  = fuzz.token_set_ratio(a_key, b_key)
        tsort = fuzz.token_sort_ratio(a_key, b_key)
        part  = fuzz.partial_ratio(a_key, b_key)
        jw    = 100.0 * RF_JW.normalized_similarity(a_key, b_key) if RF_JW else 0.0
    else:
        import difflib
        a_tok = " ".join(sorted(a_tokens))
        b_tok = " ".join(sorted(b_tokens))
        ratio = 100.0 * difflib.SequenceMatcher(None, a_tok, b_tok).ratio()
        tset = tsort = part = ratio
        jw   = 0.0

    return 0.35*tset + 0.25*tsort + 0.15*part + 0.15*jacc + 0.10*jw
