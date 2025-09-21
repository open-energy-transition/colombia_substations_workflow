# Colombia Substations Workflow

End-to-end, reproducible pipeline to ingest, clean, enrich, merge, and match substation datasets from **XM (PARATEC)**, **UPME**, and **OSM**, and produce an **interactive map**.

The workflow is orchestrated with **Snakemake**, is **cross-platform** (Windows/Linux/macOS), and writes artifacts into **per-stage output folders**.

---

## Features

- Single source input: `data/PARATEC_Subestaciones30-08-2025.xlsx` (needs to be downloaded first from https://paratec.xm.com.co/reportes/subestaciones)
- Deterministic DAG with Snakemake (`Snakefile` + `pipeline.smk.yaml`)
- Per-stage outputs: `outputs/osm`, `outputs/xm`, `outputs/upme`, `outputs/merge`, `outputs/match`, `outputs/map`
- CSV/GeoJSON artifacts for GIS + QA
- Final **interactive HTML map** for exploration

---

## Project structure

```
colombia_substations_workflow/
├─ Snakefile
├─ pipeline.smk.yaml
├─ scripts/
│  ├─ get_osm_subs.py
│  ├─ clean_paratec_subestaciones.py
│  ├─ get_subs_upme.py
│  ├─ merge_subestaciones.py
│  ├─ osm_paratec_match.py
│  └─ plot_map.py
└─ data/
   └─ PARATEC_Subestaciones30-08-2025.xlsx
```

Outputs are written under:

```
outputs/
├─ osm/
├─ xm/
├─ upme/
├─ merge/
├─ match/
└─ map/
```

---

## Prerequisites

- Python 3.9+ (tested with 3.11)
- Snakemake (e.g., `pip install snakemake`)
- Libraries used by the scripts: `pandas`, `numpy`, `requests`, `rapidfuzz` (optional), `folium`/`branca`

> Tip: create a virtualenv and install your deps with `pip install -r requirements.txt` (optional).

---

## Quick start

1. Place the XM Excel at:
   ```
   data/PARATEC_Subestaciones30-08-2025.xlsx
   ```
2. Run the full pipeline:
   ```bash
   snakemake -j 4
   ```
3. Open the final interactive map:
   ```
   outputs/map/paratec_map.html
   ```

> Cross-platform: the `Snakefile` uses Python `run:` blocks, so no Bash is required on Windows.

---

## Configuration

All paths are controlled in `pipeline.smk.yaml`:

```yaml
workdir: "."

scripts:
  get_osm_subs: "scripts/get_osm_subs.py"
  clean_paratec_subestaciones: "scripts/clean_paratec_subestaciones.py"
  get_subs_upme: "scripts/get_subs_upme.py"
  merge_subestaciones: "scripts/merge_subestaciones.py"
  osm_paratec_match: "scripts/osm_paratec_match.py"
  plot_map: "scripts/plot_map.py"

xm_excel:
  path: "data/PARATEC_Subestaciones30-08-2025.xlsx"
  expected_name: "PARATEC_Subestaciones30-08-2025.xlsx"

stage_dirs:
  osm:   "outputs/osm"
  xm:    "outputs/xm"
  upme:  "outputs/upme"
  merge: "outputs/merge"
  match: "outputs/match"
  map:   "outputs/map"

# optional: pin a specific interpreter
# python: "/usr/bin/python3"
```

If you relocate scripts or rename the Excel, update this YAML only.

---

## Workflow stages

Stages run **inside their own output folder**, so each script’s relative I/O lands neatly per stage.

### 1) OSM pull & filter → `outputs/osm/`
**Script:** `scripts/get_osm_subs.py`  
**Inputs:** _none_ (fetches from OSM/Overpass or configured source)  
**Outputs:**
- `osm_substations_filtered.csv` (main)
- `osm_substations_named_no_voltage.geojson` (QA)
- `osm_substations_voltage_no_name.geojson` (QA)

---

### 2) XM (PARATEC) clean → `outputs/xm/`
**Script:** `scripts/clean_paratec_subestaciones.py`  
**Inputs:**  
- `PARATEC_Subestaciones30-08-2025.xlsx` (copied into the stage folder under the expected filename)

**Outputs:**
- `PARATEC_substations.csv` (tidy CSV; fixes merged rows via selective forward-fill per substation and preserves in-cell line breaks)

---

### 3) UPME pull → `outputs/upme/`
**Script:** `scripts/get_subs_upme.py`  
**Inputs:** _none_ (calls UPME REST service)  
**Outputs:**
- `subestaciones_upme.geojson` (pretty-printed GeoJSON)
- `subestaciones_upme.csv` (robust quoting, newline handling, and name filtering)

---

### 4) Merge XM + UPME → `outputs/merge/`
**Script:** `scripts/merge_subestaciones.py`  
**Inputs:**
- `PARATEC_substations.csv` (from Stage 2)
- `subestaciones_upme.csv` (from Stage 3)

**Outputs:**
- **Main:** `PARATEC_with_coords.csv` (XM with coordinates from UPME where available)
- **Secondary:**
  - `PARATEC_unmatched_in_UPME.csv`
  - `PARATEC_fuzzy_matches.csv`

---

### 5) OSM ↔ XM match + reports → `outputs/match/`
**Script:** `scripts/osm_paratec_match.py`  
**Inputs:**
- `PARATEC_with_coords.csv` (from Stage 4)
- `osm_substations_filtered.csv` (from Stage 1)

**Outputs:**
- **Main:** `OSM_PARATEC_enriched.csv` (integrated OSM + XM + UPME dataset)
- **Secondary:**
  - `PARATEC_enriched_coords.csv`
  - `PARATEC_not_in_OSM.geojson`
  - `PARATEC_not_in_OSM.csv`
  - `PARATEC_not_in_OSM_missing_coords.csv`
  - `MATCHES_by_type.csv`
  - `MATCHES_summary.csv`

---

### 6) Interactive map → `outputs/map/`
**Script:** `scripts/plot_map.py`  
**Inputs:**
- `OSM_PARATEC_enriched.csv` (from Stage 5)

**Outputs:**
- `paratec_map.html` (interactive map for exploration/QA)

---

## Usage tips

- **Run everything**
  ```bash
  snakemake -j 4
  ```

- **Build a specific final target** (prerequisites build automatically)
  ```bash
  snakemake outputs/map/paratec_map.html -j 4 -p
  ```

- **Rebuild one stage from scratch** (delete that stage’s outputs, then rebuild)
  ```bash
  rm -f outputs/merge/*
  snakemake outputs/merge/PARATEC_with_coords.csv -j 2 -p
  ```

- **Visualize the DAG**
  ```bash
  snakemake --dag | dot -Tpng > dag.png
  ```

- **Pick parallelism**
  - Day-to-day: `-j 3` to `-j 6`
  - Debugging: `-j 1 -p --verbose`

- **Run only the map after matching**
  ```bash
  snakemake outputs/map/paratec_map.html -j 1 -p
  ```


- **Inspect detailed logs**
  - Run with:
    ```bash
    snakemake -j 1 -p --verbose
    ```
  - Or check `.snakemake/log/*.log`.

---
