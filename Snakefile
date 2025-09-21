# Snakefile — cross-platform orchestration with per-stage output folders.
# Stage order (1..6):
#   1) OSM pull & filter        → outputs/osm
#   2) XM (PARATEC) clean       → outputs/xm
#   3) UPME pull                → outputs/upme
#   4) Merge XM+UPME            → outputs/merge
#   5) OSM↔XM match + reports   → outputs/match
#   6) Interactive map (Folium) → outputs/map

from pathlib import Path
import shutil
import subprocess
import sys

configfile: "pipeline.smk.yaml"

WORKDIR = Path(config["workdir"]).resolve()
STAGE   = {k: Path(v).resolve() for k, v in config["stage_dirs"].items()}
SCRIPTS = config["scripts"]
XM_XLSX = Path(config["xm_excel"]["path"]).resolve()
XM_EXPECTED_NAME = config["xm_excel"]["expected_name"]
PYEXE = config.get("python", sys.executable)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def runpy(script_path: Path, workdir: Path):
    ensure_dir(workdir)
    subprocess.run([PYEXE, str(script_path.resolve())], cwd=str(workdir), check=True)

def assert_nonempty(p: Path):
    if (not p.exists()) or p.stat().st_size == 0:
        raise RuntimeError(f"Expected non-empty file not found: {p}")

# ---------------- Final targets (now includes the map) ----------------
rule all:
    input:
        # main deliverables (match stage)
        STAGE["match"] / "OSM_PARATEC_enriched.csv",
        STAGE["match"] / "PARATEC_enriched_coords.csv",
        STAGE["match"] / "PARATEC_not_in_OSM.geojson",
        STAGE["match"] / "PARATEC_not_in_OSM.csv",
        STAGE["match"] / "PARATEC_not_in_OSM_missing_coords.csv",
        STAGE["match"] / "MATCHES_by_type.csv",
        STAGE["match"] / "MATCHES_summary.csv",
        # intermediates per stage
        STAGE["osm"]   / "osm_substations_filtered.csv",
        STAGE["osm"]   / "osm_substations_named_no_voltage.geojson",
        STAGE["osm"]   / "osm_substations_voltage_no_name.geojson",
        STAGE["xm"]    / "PARATEC_substations.csv",
        STAGE["upme"]  / "subestaciones_upme.csv",
        STAGE["upme"]  / "subestaciones_upme.geojson",
        STAGE["merge"] / "PARATEC_with_coords.csv",
        STAGE["merge"] / "PARATEC_unmatched_in_UPME.csv",
        STAGE["merge"] / "PARATEC_fuzzy_matches.csv",
        # final interactive map
        STAGE["map"]   / "paratec_map.html"

# ---------------- Stage 1: OSM pull & filter → outputs/osm ----------------
rule get_osm_substations:
    output:
        csv = STAGE["osm"] / "osm_substations_filtered.csv",
        gj1 = STAGE["osm"] / "osm_substations_named_no_voltage.geojson",
        gj2 = STAGE["osm"] / "osm_substations_voltage_no_name.geojson"
    run:
        runpy(Path(SCRIPTS["get_osm_subs"]), STAGE["osm"])
        assert_nonempty(Path(output.csv))
        assert_nonempty(Path(output.gj1))
        assert_nonempty(Path(output.gj2))

# ---------------- Stage 2: XM (PARATEC) Excel → CSV → outputs/xm ----------------
rule clean_paratec_excel_to_csv:
    input:
        xlsx = XM_XLSX
    output:
        csv = STAGE["xm"] / "PARATEC_substations.csv"
    run:
        ensure_dir(STAGE["xm"])
        expected = STAGE["xm"] / XM_EXPECTED_NAME
        if str(input.xlsx) != str(expected):
            shutil.copyfile(str(input.xlsx), str(expected))
        runpy(Path(SCRIPTS["clean_paratec_subestaciones"]), STAGE["xm"])
        assert_nonempty(Path(output.csv))

# ---------------- Stage 3: UPME pull → outputs/upme ----------------
rule get_upme_substations:
    output:
        csv = STAGE["upme"] / "subestaciones_upme.csv",
        gj  = STAGE["upme"] / "subestaciones_upme.geojson"
    run:
        runpy(Path(SCRIPTS["get_subs_upme"]), STAGE["upme"])
        assert_nonempty(Path(output.csv))
        assert_nonempty(Path(output.gj))

# ---------------- Stage 4: Merge XM + UPME → outputs/merge ----------------
rule merge_paratec_upme:
    input:
        par_csv  = STAGE["xm"]   / "PARATEC_substations.csv",
        upme_csv = STAGE["upme"] / "subestaciones_upme.csv"
    output:
        merged    = STAGE["merge"] / "PARATEC_with_coords.csv",
        unmatched = STAGE["merge"] / "PARATEC_unmatched_in_UPME.csv",
        fuzzy     = STAGE["merge"] / "PARATEC_fuzzy_matches.csv"
    run:
        ensure_dir(STAGE["merge"])
        shutil.copyfile(str(input.par_csv),  str(STAGE["merge"] / "PARATEC_substations.csv"))
        shutil.copyfile(str(input.upme_csv), str(STAGE["merge"] / "subestaciones_upme.csv"))
        runpy(Path(SCRIPTS["merge_subestaciones"]), STAGE["merge"])
        assert_nonempty(Path(output.merged))
        assert_nonempty(Path(output.unmatched))
        assert_nonempty(Path(output.fuzzy))

# ---------------- Stage 5: OSM ↔ XM match + reports → outputs/match ----------------
rule osm_paratec_match:
    input:
        merged = STAGE["merge"] / "PARATEC_with_coords.csv",
        osm    = STAGE["osm"]   / "osm_substations_filtered.csv"
    output:
        enr       = STAGE["match"] / "OSM_PARATEC_enriched.csv",
        par_enr   = STAGE["match"] / "PARATEC_enriched_coords.csv",
        not_gj    = STAGE["match"] / "PARATEC_not_in_OSM.geojson",
        not_csv   = STAGE["match"] / "PARATEC_not_in_OSM.csv",
        not_miss  = STAGE["match"] / "PARATEC_not_in_OSM_missing_coords.csv",
        m_type    = STAGE["match"] / "MATCHES_by_type.csv",
        m_sum     = STAGE["match"] / "MATCHES_summary.csv"
    run:
        ensure_dir(STAGE["match"])
        shutil.copyfile(str(input.merged), str(STAGE["match"] / "PARATEC_with_coords.csv"))
        shutil.copyfile(str(input.osm),    str(STAGE["match"] / "osm_substations_filtered.csv"))
        runpy(Path(SCRIPTS["osm_paratec_match"]), STAGE["match"])
        assert_nonempty(Path(output.enr))
        assert_nonempty(Path(output.par_enr))
        assert_nonempty(Path(output.not_gj))
        assert_nonempty(Path(output.not_csv))
        assert_nonempty(Path(output.not_miss))
        assert_nonempty(Path(output.m_type))
        assert_nonempty(Path(output.m_sum))

# ---------------- Stage 6: Interactive map (Folium) → outputs/map ----------------
rule plot_interactive_map:
    input:
        enr = STAGE["match"] / "OSM_PARATEC_enriched.csv"
    output:
        html = STAGE["map"] / "paratec_map.html"
    run:
        ensure_dir(STAGE["map"])
        # plot_map.py expects 'OSM_PARATEC_enriched.csv' in CWD and creates 'paratec_map.html'
        shutil.copyfile(str(input.enr), str(STAGE["map"] / "OSM_PARATEC_enriched.csv"))
        runpy(Path(SCRIPTS["plot_map"]), STAGE["map"])
        assert_nonempty(Path(output.html))
