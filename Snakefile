# Snakefile — final, no input copying
configfile: "config.yaml"

from pathlib import Path
import os, subprocess, sys

D = config["dirs"]
S = config["steps"]
SCR = config["scripts"]
F = config["files"]
INP = config["inputs"]

DATA = Path(D["data"]).resolve()
SCRIPTS = Path(D["scripts"]).resolve()
OUT = Path(D["outputs"]).resolve()

# Per-step folders
D01 = OUT / S["osm"]
D02 = OUT / S["paratec_clean"]
D03 = OUT / S["xm_dashboard"]
D04 = OUT / S["upme"]
D05 = OUT / S["xm_upme_union"]
D06 = OUT / S["enrich_xm"]
D07 = OUT / S["osm_xmparatec"]
D08 = OUT / S["plot"]

rule all:
    input:
        D07 / F["osm_xmparatec"]["paratec_enriched_osm"],
        D08 / F["plot"]["html"]

# Helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_py_with_args(script, cwd, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SCRIPTS)  # so "import matching_utils" works
    subprocess.run([sys.executable, str(script), *args], cwd=str(cwd), check=True, env=env)

###############################################################################
# 1) OSM (no upstream inputs; script controls country internally)
###############################################################################
rule osm_substations:
    output:
        D01 / F["osm"]["dedup"],
        D01 / F["osm"]["raw"],
        D01 / F["osm"]["filtered"],
        D01 / F["osm"]["dropped"],
    run:
        ensure_dir(D01)
        run_py_with_args(SCRIPTS / SCR["osm"], D01, [])

###############################################################################
# 2) Clean PARATEC Excel -> CSV (reads from data/, no upstream outputs)
###############################################################################
rule clean_paratec_xlsx:
    input:
        xlsx = DATA / INP["paratec_xlsx"]
    output:
        D02 / F["paratec_clean"]["csv"]
    run:
        ensure_dir(D02)
        # Stage the Excel where the script expects it (CWD)
        from pathlib import Path
        import shutil
        shutil.copy2(str(input.xlsx), str(D02 / Path(input.xlsx).name))
        # Run the original script in the step directory
        run_py_with_args(SCRIPTS / SCR["clean_paratec"], D02, [])
        assert Path(output[0]).exists(), f"Expected output missing: {output[0]}"


###############################################################################
# 3) XM dashboard dump (no upstream inputs)
###############################################################################
rule xm_dashboard_markers:
    output:
        D03 / F["xm_dashboard"]["markers"],
        D03 / F["xm_dashboard"]["lines"]
    run:
        ensure_dir(D03)
        run_py_with_args(SCRIPTS / SCR["xm_dashboard"], D03, [])

###############################################################################
# 4) UPME download (no upstream inputs)
###############################################################################
rule upme_substations:
    output:
        D04 / F["upme"]["geojson"],
        D04 / F["upme"]["csv"]
    run:
        ensure_dir(D04)
        run_py_with_args(SCRIPTS / SCR["upme"], D04, [])

###############################################################################
# 5) XM ⟵ UPME union 
###############################################################################
rule xm_upme_union:
    input:
        xm    = D03 / F["xm_dashboard"]["markers"],
        upme  = D04 / F["upme"]["csv"]
    output:
        D05 / F["xm_upme_union"]["enriched_xm"]
    run:
        ensure_dir(D05)
        run_py_with_args(
            SCRIPTS / "xm_upme_union_cli.py",
            D05,
            [
              "--xm", str(input.xm),
              "--upme", str(input.upme),
              "--out-enriched", str(output[0])
            ]
        )

###############################################################################
# 6) Enrich PARATEC with XM coords 
###############################################################################
rule paratec_enrich_with_xm:
    input:
        paratec = D02 / F["paratec_clean"]["csv"],
        xm      = D05 / F["xm_upme_union"]["enriched_xm"]
    output:
        D06 / F["enrich_xm"]["paratec_enriched_xm"]
    run:
        ensure_dir(D06)
        run_py_with_args(
            SCRIPTS / "enrich_xm_with_coords_cli.py",
            D06,
            [
              "--paratec", str(input.paratec),
              "--xm",      str(input.xm),
              "--out-enriched", str(output[0])
            ]
        )

###############################################################################
# 7) OSM ⟷ PARATEC union 
###############################################################################
rule osm_xmparatec_union:
    input:
        osm     = D01 / F["osm"]["dedup"],
        paratec = D06 / F["enrich_xm"]["paratec_enriched_xm"]
    output:
        D07 / F["osm_xmparatec"]["paratec_enriched_osm"],
        D07 / F["osm_xmparatec"]["osm_unmatched"]
    run:
        ensure_dir(D07)
        run_py_with_args(
            SCRIPTS / "osm_xmparatec_cli.py",
            D07,
            [
              "--osm",     str(input.osm),
              "--paratec", str(input.paratec),
              "--out-enriched-location", str(output[0]),
              "--out-unmatched-sites",   str(output[1])
            ]
        )

###############################################################################
# 8) Plot final map
###############################################################################
rule plot_map:
    input:
        csv = D07 / F["osm_xmparatec"]["paratec_enriched_osm"]
    output:
        D08 / F["plot"]["html"]
    run:
        ensure_dir(D08)
        run_py_with_args(
            SCRIPTS / "plot_map_cli.py",
            D08,
            [
              "--in",  str(input.csv),
              "--out", str(output[0])
            ]
        )
