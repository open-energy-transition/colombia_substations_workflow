# Colombia Substations Workflow

End-to-end, reproducible pipeline to ingest, clean, enrich, match and visualize **substations** from **XM (PARATEC)**, **UPME**, and **OSM**, producing a final **interactive map**.

- Cross‑platform (Windows / macOS / Linux) — uses Python-based Snakemake rules (no bash needed on Windows).
- Clean folder layout: `data/`, `scripts/`, `outputs/<step>/...`.
- No duplication of inputs between steps — scripts receive absolute paths via CLI (for steps 5–8).

---

## Project layout

```
.
├─ Snakefile
├─ config.yaml
├─ data/
│  └─ PARATEC_Subestaciones30-08-2025.xlsx     # your downloaded Excel (required)
├─ scripts/
│  ├─ get_osm_subs_improved.py                  # step 1 (country set inside script)
│  ├─ clean_paratec_subestaciones.py            # step 2
│  ├─ get_subs_xm_dashboard.py                  # step 3
│  ├─ get_subs_upme.py                          # step 4
│  ├─ xm_upme_union_cli.py                      # step 5 
│  ├─ enrich_xm_with_coords_cli.py              # step 6 
│  ├─ osm_xmparatec_cli.py                      # step 7 
│  ├─ plot_map_cli.py                           # step 8 
│  └─ matching_utils.py                         # shared helpers 
└─ outputs/
   ├─ 01_osm/
   ├─ 02_paratec_clean/
   ├─ 03_xm_dashboard/
   ├─ 04_upme/
   ├─ 05_xm_upme_union/
   ├─ 06_enrich_xm/
   ├─ 07_osm_xmparatec/
   └─ 08_plot/
```

> The `outputs/` subfolders are created automatically by Snakemake on first run.

---

## Requirements

- **Python** 3.9+ (tested with 3.11)
- **Snakemake** (`pip install snakemake`)
- Script deps: `pandas`, `requests`, `rapidfuzz` (if you use it in your utils), `folium`, `branca`

Optional (recommended):
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt   # create one if you like
```

---

## Configuration (`config.yaml`)

```yaml
dirs:
  data: "data"
  scripts: "scripts"
  outputs: "outputs"

steps:
  osm: "01_osm"
  paratec_clean: "02_paratec_clean"
  xm_dashboard: "03_xm_dashboard"
  upme: "04_upme"
  xm_upme_union: "05_xm_upme_union"
  enrich_xm: "06_enrich_xm"
  osm_xmparatec: "07_osm_xmparatec"
  plot: "08_plot"

scripts:
  osm: "get_osm_subs_improved.py"
  clean_paratec: "clean_paratec_subestaciones.py"
  xm_dashboard: "get_subs_xm_dashboard.py"
  upme: "get_subs_upme.py"
  xm_upme_union: "xm_upme_union.py"           
  enrich_xm_with_coords: "enrich_xm_with_coords.py"  
  osm_xmparatec: "osm_xmparatec.py"                  
  plot_map: "plot_map.py"                            
  utils: "matching_utils.py"

files:
  osm:
    raw: "osm_substations_raw.csv"
    filtered: "osm_substations_filtered.csv"
    dropped: "osm_substations_dropped_debug.csv"
    dedup: "osm_substations_dedup.csv"
  paratec_clean:
    csv: "PARATEC_substations.csv"
  xm_dashboard:
    markers: "getMarkers.csv"
    lines: "getLines.csv"
  upme:
    geojson: "subestaciones_upme.geojson"
    csv: "subestaciones_upme.csv"
  xm_upme_union:
    enriched_xm: "XM_enriched_with_UPME_coords.csv"
  enrich_xm:
    paratec_enriched_xm: "PARATEC_enriched_with_XMcoords.csv"
  osm_xmparatec:
    paratec_enriched_osm: "PARATEC_enriched_with_OSMcoords_with_location.csv"
    osm_unmatched: "OSM_unmatched_sites.csv"
  plot:
    html: "paratec_map.html"

inputs:
  paratec_xlsx: "PARATEC_Subestaciones30-08-2025.xlsx"

```

---

## Running

**1) Put the Excel in `data/`:**
```
data/PARATEC_Subestaciones30-08-2025.xlsx
```

**2) Run the full workflow:**
```bash
snakemake -j 4
```

**3) Open the final map:**
```
outputs/08_plot/paratec_map.html
```


---

## Workflow stages

Each stage runs in its own output folder and writes its own artifacts there.

### 1) OSM pull & dedup → `outputs/01_osm/`
**Script:** `scripts/get_osm_subs_improved.py`  
**Inputs:** none (Overpass)  
**Outputs:**
- `osm_substations_dedup.csv` (main, location‑based dedup)
- `osm_substations_filtered.csv` (named only, post‑dedup)
- `osm_substations_raw.csv`, `osm_substations_dropped_debug.csv`

### 2) Clean PARATEC Excel → `outputs/02_paratec_clean/`
**Script:** `scripts/clean_paratec_subestaciones.py`  
**Input:** `data/PARATEC_Subestaciones30-08-2025.xlsx`  
**Output:** `PARATEC_substations.csv`

> This original script expects the downloaded Excel by a fixed name; the Snakefile stages it into the step dir before running (only step with staging). It could be also CLI‑enable later to avoid staging entirely.

### 3) XM dashboard (markers/lines) → `outputs/03_xm_dashboard/`
**Script:** `scripts/get_subs_xm_dashboard.py`  
**Outputs:** `getMarkers.csv`, `getLines.csv`

### 4) UPME pull → `outputs/04_upme/`
**Script:** `scripts/get_subs_upme.py`  
**Outputs:** `subestaciones_upme.geojson`, `subestaciones_upme.csv`

### 5) XM ⟵ UPME union → `outputs/05_xm_upme_union/`
**Script:** `scripts/xm_upme_union_cli.py`  
**Inputs:** step 3 markers, step 4 UPME CSV  
**Output:** `XM_enriched_with_UPME_coords.csv`

### 6) Enrich PARATEC with XM coords → `outputs/06_enrich_xm/`
**Script:** `scripts/enrich_xm_with_coords_cli.py`  
**Inputs:** step 2 PARATEC CSV, step 5 enriched XM  
**Output:** `PARATEC_enriched_with_XMcoords.csv`

### 7) OSM ⟷ PARATEC union + reports → `outputs/07_osm_xmparatec/`
**Script:** `scripts/osm_xmparatec_cli.py`  
**Inputs:** step 1 OSM dedup, step 6 PARATEC_enriched_with_XMcoords  
**Outputs:**  
- `PARATEC_enriched_with_OSMcoords_with_location.csv` (main)  
- `OSM_unmatched_sites.csv` (report)  
- (script also logs coverage stats and optional GeoJSONs)

### 8) Interactive map → `outputs/08_plot/`
**Script:** `scripts/plot_map_cli.py`  
**Input:** step 7 curated CSV  
**Output:** `paratec_map.html`

---

## Handy commands

- Run a specific step (builds prerequisites automatically):
  ```bash
  snakemake outputs/08_plot/paratec_map.html -j 2 -p
  ```

- Visualize the DAG:
  ```bash
  snakemake --dag | dot -Tpng > dag.png
  ```

- Force rerun a step:
  ```bash
  rm -f outputs/05_xm_upme_union/*
  snakemake outputs/05_xm_upme_union/XM_enriched_with_UPME_coords.csv -j 2 -p
  ```

- Increase parallel jobs:
  ```bash
  snakemake -j 6
  ```

---

## Troubleshooting

- **`clean_paratec_subestaciones.py` can’t find the Excel**  
  Ensure the file exists at `data/PARATEC_Subestaciones30-08-2025.xlsx`. The Snakefile stages it into the step folder before running.


- **Overpass rate limits**  
  If OSM requests time out, simply re-run Snakemake. The OSM step has no inputs, so it will re-execute and write fresh outputs.


---

## License & attribution

This repository is open-source software developed by [Open Energy Transition (OET)](https://openenergytransition.org)
and distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

You are free to use, modify, and distribute this software under the same license,
provided that any derivative works or services built upon it remain open source.

