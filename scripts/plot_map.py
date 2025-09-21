#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folium map of substations (from OSM_PARATEC_enriched.csv) and OSM power lines in Colombia.

- Substations are from CSV (lat/lon, Nombre, voltage, SC capacity, connected lines).
- Power lines are fetched from OSM with 'out body; >; out skel qt;'.
- Clicking a substation shows Name, Voltage, SC capacity, Connected lines.
- Lines + substations styled with vivid voltage colors (distinct color per voltage level).
- Dark background map.
- Legend only shows voltages present in the CSV (and/or lines).

Output:
  - paratec_map.html

Requirements:
    pip install requests pandas folium branca
"""

import re
import pandas as pd
import requests
import folium
from branca.element import MacroElement, Template

CSV_NAME = "OSM_PARATEC_enriched.csv"
OUTPUT_HTML = "paratec_map.html"
COUNTRY_ALPHA2 = "CO"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Fixed vivid colors for common voltage levels (kV) to ensure consistency between runs
PREFERRED_VOLTAGE_COLORS = {
    34:  "#00E676",   # vivid green
    44:  "#00B0FF",   # vivid sky
    66:  "#7C4DFF",   # deep violet
    110: "#00E5FF",   # cyan
    115: "#2979FF",   # strong blue
    138: "#AA00FF",   # magenta
    220: "#FF6D00",   # vivid orange
    230: "#00C853",   # emerald
    345: "#FFD600",   # bright yellow
    400: "#D50000",   # vivid red (dark tone)
    500: "#FF1744",   # hot red
    750: "#FF4081",   # pink
}

VIVID_CYCLE = [
    "#00BCD4", "#FF5722", "#8BC34A", "#3F51B5",
    "#FFC107", "#E91E63", "#4CAF50", "#9C27B0",
    "#FF9800", "#2196F3", "#CDDC39", "#F44336",
]

def parse_voltage(v):
    """
    Parse voltage field into numeric kV (int).
    - Accepts strings like "230", "230.0", "230000", "110;220", "110/220", etc.
    - If values look like volts (e.g., 230000), convert to kV.
    - If multiple numbers present, returns the HIGHEST (most relevant for transmission).
    """
    if pd.isna(v):
        return None
    s = str(v).replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    values = []
    for n in nums:
        val = float(n)
        # Heuristic: if over ~1000 assume it's volts, convert to kV
        if val >= 1000.0:
            val = val / 1000.0
        values.append(val)
    if not values:
        return None
    return int(round(max(values)))  # choose the highest voltage level

def safe_val(v):
    """Return value as string, or 'n/a' if missing."""
    if pd.isna(v) or str(v).strip().lower() in ["nan", "none", ""]:
        return "n/a"
    return str(v)

def add_legend(map_obj, voltage_colors):
    # Sorted by numeric kV
    items = "".join(
        f'<div style="line-height:1.2em;"><span style="color:{c}; font-weight:bold;">&#9632;</span> {kv} kV</div>'
        for kv, c in sorted(voltage_colors.items(), key=lambda x: x[0])
    )
    legend_html = f"""
{{% macro html(this, kwargs) %}}
<div style="
  position: fixed;
  bottom: 20px; left: 20px;
  z-index: 9999;
  background: rgba(0,0,0,0.85);
  color: white;
  padding: 10px 12px;
  border: 1px solid #888;
  border-radius: 6px;
  font-size: 13px;
  max-width: 220px;
">
<div style="font-weight:700; margin-bottom:6px;">Voltage Legend</div>
{items}
</div>
{{% endmacro %}}
"""
    macro = MacroElement()
    macro._template = Template(legend_html)
    map_obj.get_root().add_child(macro)

def build_voltage_palette(df_kv, lines_kv):
    """
    Build an ordered voltage→color mapping based on:
      1) Fixed colors for common voltages (PREFERRED_VOLTAGE_COLORS)
      2) Vivid cycle for any remaining voltages encountered
    Ensures 1 color per distinct level seen in either substations or lines.
    """
    present = set(k for k in df_kv if k is not None) | set(k for k in lines_kv if k is not None)
    present = sorted(present)  # deterministic order

    palette = {}
    # First assign preferred colors where applicable
    for kv in present:
        if kv in PREFERRED_VOLTAGE_COLORS:
            palette[kv] = PREFERRED_VOLTAGE_COLORS[kv]

    # Then assign a vivid cycle for the rest
    cycle_idx = 0
    for kv in present:
        if kv not in palette:
            palette[kv] = VIVID_CYCLE[cycle_idx % len(VIVID_CYCLE)]
            cycle_idx += 1

    return palette

def fetch_osm_lines(country_alpha2):
    """Fetch power lines for the given ISO3166-1 alpha-2 using Overpass."""
    ql = f"""
[out:json][timeout:300];
area["ISO3166-1"="{country_alpha2}"]->.searchArea;
(
  way["power"="line"](area.searchArea);
  way["power"="minor_line"](area.searchArea);
);
out body;
>;
out skel qt;
"""
    r = requests.post(OVERPASS_URL, data={"data": ql}, timeout=400)
    r.raise_for_status()
    return r.json()

def main():
    # ---------------- Substations ----------------
    df = pd.read_csv(CSV_NAME, dtype=str, encoding="utf-8")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    name_col   = "Nombre"
    volt_col   = "Voltaje nominal de operación [kV]"
    scc_col    = "Capacidad de cortocircuito [kA]"
    lineas_col = "Líneas"

    for col in (name_col, volt_col, scc_col, lineas_col):
        if col not in df.columns:
            raise SystemExit(f"CSV must contain column '{col}'")

    # Drop rows without coordinates
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Extract numeric kV for each substation
    df["kv"] = df[volt_col].map(parse_voltage)
    kv_substations = list(df["kv"].dropna().astype(int).unique())
    print(f"Loaded {len(df)} substations; voltages found (substations): {sorted(kv_substations)}")

    # ---------------- Power lines from OSM ----------------
    print("Fetching OSM lines...")
    osm = fetch_osm_lines(COUNTRY_ALPHA2)

    # Build node dictionary
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm.get("elements", []) if el.get("type") == "node"}

    # Build ways, parse voltage (pick highest), keep coords
    lines = []
    lines_kv_levels = set()
    for el in osm.get("elements", []):
        if el.get("type") == "way":
            node_ids = el.get("nodes", [])
            coords = [nodes[nid] for nid in node_ids if nid in nodes]
            tags = el.get("tags", {}) or {}
            v_tag = tags.get("voltage", "")
            kv = parse_voltage(v_tag)
            if kv is not None:
                lines_kv_levels.add(kv)
            if coords:
                lines.append({"kv": kv, "coords": coords})
    print(f"Reconstructed {len(lines)} OSM lines; voltages found (lines): {sorted(lines_kv_levels)}")

    # ---------------- Build voltage palette ----------------
    voltage_colors = build_voltage_palette(kv_substations, lines_kv_levels)
    print(f"Voltage color mapping: { {k: v for k, v in sorted(voltage_colors.items())} }")

    # ---------------- Folium map ----------------
    center = [df["lat"].mean(), df["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB Dark_Matter")

    # Substations with popup (vivid fill + dark outline for contrast)
    for _, row in df.iterrows():
        kv_val = safe_val(row.get(volt_col))
        kv_num = parse_voltage(row.get(volt_col))
        color  = voltage_colors.get(kv_num, "#BDBDBD")  # light gray fallback

        popup_html = f"""
        <b>Name:</b> {safe_val(row[name_col])}<br>
        <b>Voltage:</b> {kv_val} kV<br>
        <b>SC capacity:</b> {safe_val(row[scc_col])} kA<br>
        <b>Connected lines:</b> {safe_val(row[lineas_col])}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="#111111",           # dark outline for contrast
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"{safe_val(row[name_col])} ({kv_val} kV)",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fmap)

    # Lines (use same palette; slightly thicker for higher voltages)
    for ln in lines:
        kv = ln["kv"]
        color = voltage_colors.get(kv, "#757575")  # medium gray fallback
        weight = 2.5 if (kv is None or kv < 200) else (3.5 if kv < 400 else 4.5)
        folium.PolyLine(
            locations=ln["coords"],
            color=color,
            weight=weight,
            opacity=0.9,
            tooltip=f"Line {kv} kV" if kv else "Line (no voltage)",
        ).add_to(fmap)

    add_legend(fmap, voltage_colors)
    fmap.save(OUTPUT_HTML)
    print(f"Saved map to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
