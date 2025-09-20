# clean_paratec_subestaciones_selective_ffill.py
# Convierte el XLSX a CSV y solo hace forward-fill en celdas originalmente vacías,
# sin tocar los '-' reales. Propaga dentro de cada subestación (grupo por 'Nombre').

import pandas as pd
from pathlib import Path
import csv

INPUT_FILE  = Path("PARATEC_Subestaciones30-08-2025.xlsx")
OUTPUT_FILE = Path("PARATEC_substations.csv")

def main():
    # 1) Leer Excel con la segunda fila como encabezados, todo como texto
    df = pd.read_excel(INPUT_FILE, header=1, dtype=str)

    # 2) Normalizar blancos: convertir cadenas vacías a NaN (preserva '-')
    def normalize_blank(x):
        if isinstance(x, str) and x.strip() == "":
            return pd.NA
        return x
    df = df.applymap(normalize_blank)

    # 3) Asegurar que 'Nombre' defina el grupo (arregla celdas combinadas)
    if "Nombre" in df.columns:
        df["Nombre"] = df["Nombre"].ffill()

    # 4) Forward-fill selectivo por subestación:
    #    - Calcula una versión ffill() por grupo
    #    - Solo rellena donde el valor ORIGINAL era NaN (no reemplaza '-')
    if "Nombre" in df.columns:
        def selective_ffill(g: pd.DataFrame) -> pd.DataFrame:
            filled = g.ffill()
            return g.where(~g.isna(), filled)
        df = df.groupby("Nombre", group_keys=False).apply(selective_ffill)

    # 5) Rellenar NaN reales que queden con '-'
    df = df.fillna("-")

    # 6) Exportar CSV con ; y BOM, preservando saltos de línea en celdas
    df.to_csv(
        OUTPUT_FILE,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        quoting=csv.QUOTE_NONNUMERIC
    )

    print(f"Exportado: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
