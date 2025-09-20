# Snakefile (Windows-safe) — pure Python run blocks, no bash utils

from pathlib import Path
import shutil
import subprocess
import sys

configfile: "pipeline.smk.yaml"

WORKDIR = Path(config["workdir"]).resolve()
SCRIPTS = config["scripts"]
XM_XLSX = Path(config["xm_excel"]["path"]).resolve()
XM_EXPECTED_NAME = config["xm_excel"]["expected_name"]
PYEXE = config.get("python", sys.executable)

def runpy(script_path: Path, workdir: Path):
    subprocess.run([PYEXE, str(script_path)], cwd=str(workdir), check=True)

def assert_nonempty(p: Path):
    if (not p.exists()) or p.stat().st_size == 0:
        raise RuntimeError(f"Expected non-empty file not found: {p}")

rule all:
    input:
        WORKDIR / "OSM_PARATEC_enriched.csv",
        WORKDIR / "PARATEC_enriched_coords.csv",
        WORKDIR / "PARATEC_not_in_OSM.geojson",
        WORKDIR / "PARATEC_not_in_OSM.csv",
        WORKDIR / "PARATEC_not_in_OSM_missing_coords.csv",
        WORKDIR / "MATCHES_by_type.csv",
        WORKDIR / "MATCHES_summary.csv",
        WORKDIR / "osm_substations_filtered.csv",
        WORKDIR / "osm_substations_named_no_voltage.geojson",
        WORKDIR / "osm_substations_voltage_no_name.geojson",
        WORKDIR / "PARATEC_substations.csv",
        WORKDIR / "subestaciones_upme.csv",
        WORKDIR / "subestaciones_upme.geojson",
        WORKDIR / "PARATEC_with_coords.csv",
        WORKDIR / "PARATEC_unmatched_in_UPME.csv",
        WORKDIR / "PARATEC_fuzzy_matches.csv"

# Stage 2: XM Excel -> PARATEC_substations.csv
rule clean_paratec_excel_to_csv:
    input:
        xlsx = XM_XLSX
    output:
        csv = WORKDIR / "PARATEC_substations.csv"
    run:
        # Ensure the cleaner sees the filename it expects, in WORKDIR
        expected = WORKDIR / XM_EXPECTED_NAME
        if str(input.xlsx) != str(expected):
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(input.xlsx), str(expected))
        runpy(Path(SCRIPTS["clean_paratec_subestaciones"]).resolve(), WORKDIR)
        assert_nonempty(Path(output.csv))

# Stage 3: UPME pull
rule get_upme_substations:
    output:
        csv = WORKDIR / "subestaciones_upme.csv",
        gj  = WORKDIR / "subestaciones_upme.geojson"
    run:
        runpy(Path(SCRIPTS["get_subs_upme"]).resolve(), WORKDIR)
        assert_nonempty(Path(output.csv))
        assert_nonempty(Path(output.gj))

# Stage 1: OSM pull & filter
rule get_osm_substations:
    output:
        csv = WORKDIR / "osm_substations_filtered.csv",
        gj1 = WORKDIR / "osm_substations_named_no_voltage.geojson",
        gj2 = WORKDIR / "osm_substations_voltage_no_name.geojson"
    run:
        runpy(Path(SCRIPTS["get_osm_subs"]).resolve(), WORKDIR)
        assert_nonempty(Path(output.csv))
        assert_nonempty(Path(output.gj1))
        assert_nonempty(Path(output.gj2))

# Stage 4: Merge PARATEC + UPME
rule merge_paratec_upme:
    input:
        par_csv = WORKDIR / "PARATEC_substations.csv",
        upme_csv = WORKDIR / "subestaciones_upme.csv"
    output:
        merged   = WORKDIR / "PARATEC_with_coords.csv",
        unmatched = WORKDIR / "PARATEC_unmatched_in_UPME.csv",
        fuzzy    = WORKDIR / "PARATEC_fuzzy_matches.csv"
    run:
        runpy(Path(SCRIPTS["merge_subestaciones"]).resolve(), WORKDIR)
        assert_nonempty(Path(output.merged))
        assert_nonempty(Path(output.unmatched))
        assert_nonempty(Path(output.fuzzy))

# Stage 5: OSM ↔ PARATEC match & enriched outputs
rule osm_paratec_match:
    input:
        merged = WORKDIR / "PARATEC_with_coords.csv",
        osm    = WORKDIR / "osm_substations_filtered.csv"
    output:
        enr       = WORKDIR / "OSM_PARATEC_enriched.csv",
        par_enr   = WORKDIR / "PARATEC_enriched_coords.csv",
        not_gj    = WORKDIR / "PARATEC_not_in_OSM.geojson",
        not_csv   = WORKDIR / "PARATEC_not_in_OSM.csv",
        not_miss  = WORKDIR / "PARATEC_not_in_OSM_missing_coords.csv",
        m_type    = WORKDIR / "MATCHES_by_type.csv",
        m_sum     = WORKDIR / "MATCHES_summary.csv"
    run:
        runpy(Path(SCRIPTS["osm_paratec_match"]).resolve(), WORKDIR)
        assert_nonempty(Path(output.enr))
        assert_nonempty(Path(output.par_enr))
        assert_nonempty(Path(output.not_gj))
        assert_nonempty(Path(output.not_csv))
        assert_nonempty(Path(output.not_miss))
        assert_nonempty(Path(output.m_type))
        assert_nonempty(Path(output.m_sum))
