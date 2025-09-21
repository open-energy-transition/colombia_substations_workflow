#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Folium map of substations (from OSM_PARATEC_enriched.csv) and OSM power lines in Colombia.

- Substations are from CSV (lat/lon, Nombre, voltage, SC capacity, connected lines).
- Power lines are fetched from OSM with 'out body; >; out skel qt;'.
- Clicking a substation shows Name, Voltage, SC capacity, Connected lines.
- Lines + substations styled with vivid voltage colors.
- Dark background map.
- Legend only shows voltages present in the CSV.

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

# Vivid palette to cycle through for voltages
VIVID_COLORS = ["yellow", "limegreen", "deepskyblue", "blue",
                "orange", "red", "magenta", "cyan"]

def parse_voltage(v):
    """Parse voltage field into numeric kV."""
    if pd.isna(v):
        return None
    s = str(v).replace(",", ".")
    m = re.findall(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    val = float(m[0])
    if val > 1000 and val % 1 == 0:
        val = val / 1000.0
    return round(val)

def safe_val(v):
    """Return value as string, or 'n/a' if missing."""
    if pd.isna(v) or str(v).strip().lower() in ["nan", "none", ""]:
        return "n/a"
    return str(v)

def add_legend(map_obj, voltage_colors):
    items = "".join(
        f'<div><span style="color:{c};">&#8212;</span> {kv} kV</div>'
        for kv, c in voltage_colors.items()
    )
    legend_html = f"""
{{% macro html(this, kwargs) %}}
<div style="
  position: fixed;
  bottom: 20px; left: 20px;
  z-index: 9999;
  background: rgba(0,0,0,0.85);
  color: white;
  padding: 10px;
  border: 1px solid #888;
  border-radius: 6px;
  font-size: 13px;
">
<b>Voltage Legend</b><br>
{items}
</div>
{{% endmacro %}}
"""
    macro = MacroElement()
    macro._template = Template(legend_html)
    map_obj.get_root().add_child(macro)

def main():
    # ---------------- Substations ----------------
    df = pd.read_csv(CSV_NAME, dtype=str, encoding="utf-8")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    name_col = "Nombre"
    volt_col = "Voltaje nominal de operación [kV]"
    scc_col = "Capacidad de cortocircuito [kA]"
    lineas_col = "Líneas"

    for col in [name_col, volt_col, scc_col, lineas_col]:
        if col not in df.columns:
            raise SystemExit(f"CSV must contain column '{col}'")

    df["kv"] = df[volt_col].map(parse_voltage)
    df = df.dropna(subset=["lat", "lon"])
    print(f"Loaded {len(df)} substations from CSV")

    # Build voltage → color map dynamically with vivid colors
    kv_set = sorted({int(kv) for kv in df["kv"].dropna().unique()})
    voltage_colors = {kv: VIVID_COLORS[i % len(VIVID_COLORS)] for i, kv in enumerate(kv_set)}
    print(f"Voltages in CSV: {kv_set}")

    # ---------------- Power lines from OSM ----------------
    ql = f"""
[out:json][timeout:300];
area["ISO3166-1"="{COUNTRY_ALPHA2}"]->.searchArea;
(
  way["power"="line"](area.searchArea);
  way["power"="minor_line"](area.searchArea);
);
out body;
>;
out skel qt;
"""
    print("Fetching OSM lines...")
    r = requests.post(OVERPASS_URL, data={"data": ql}, timeout=400)
    r.raise_for_status()
    osm = r.json()

    # Build node dictionary
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}

    # Build ways
    lines = []
    for el in osm["elements"]:
        if el["type"] == "way":
            coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
            v = parse_voltage(el.get("tags", {}).get("voltage", ""))
            if coords:
                lines.append({"kv": v, "coords": coords})
    print(f"Reconstructed {len(lines)} OSM lines")

    # ---------------- Folium map ----------------
    center = [df["lat"].mean(), df["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB Dark_Matter")

    # Substations with popup
    for _, row in df.iterrows():
        kv_val = safe_val(row.get(volt_col))
        kv_num = parse_voltage(row.get(volt_col))
        color = voltage_colors.get(kv_num, "white")

        popup_html = f"""
        <b>Name:</b> {safe_val(row[name_col])}<br>
        <b>Voltage:</b> {kv_val} kV<br>
        <b>SC capacity:</b> {safe_val(row[scc_col])} kA<br>
        <b>Connected lines:</b> {safe_val(row[lineas_col])}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.95,
            tooltip=f"{safe_val(row[name_col])} ({kv_val} kV)",
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fmap)

    # Lines
    for ln in lines:
        color = voltage_colors.get(ln["kv"], "gray")
        folium.PolyLine(
            locations=ln["coords"],
            color=color,
            weight=2.5,
            opacity=0.8,
            tooltip=f"Line {ln['kv']} kV" if ln["kv"] else "Line (no voltage)"
        ).add_to(fmap)

    add_legend(fmap, voltage_colors)
    folium.LayerControl().add_to(fmap)
    fmap.save(OUTPUT_HTML)
    print(f"Saved map to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
