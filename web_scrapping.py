import requests
import pandas as pd
from tqdm import tqdm
import time

wca_id = "2023GENG02"
results_url = f"https://www.worldcubeassociation.org/api/v0/persons/{wca_id}/results"
response = requests.get(results_url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)

    if not df.empty and "competition_id" in df.columns:
        # Obtener lista única de competition_ids
        comp_ids = df["competition_id"].unique()

        # Crear diccionario con info de cada competencia
        comp_info = {}

        print("🔍 Descargando datos de competencias...")
        for cid in tqdm(comp_ids):
            comp_url = f"https://www.worldcubeassociation.org/api/v0/competitions/{cid}"
            try:
                comp_res = requests.get(comp_url)
                if comp_res.status_code == 200:
                    comp_data = comp_res.json()
                    comp_info[cid] = {
                        "competition_name": comp_data.get("name"),
                        "start_date": comp_data.get("start_date")
                    }
                else:
                    print(f"⚠️ Error al cargar competencia {cid}: {comp_res.status_code}")
                    comp_info[cid] = {
                        "competition_name": None,
                        "start_date": None
                    }
                time.sleep(0.1)  # evitar saturar la API
            except Exception as e:
                print(f"❌ Excepción en {cid}: {e}")
                comp_info[cid] = {
                    "competition_name": None,
                    "start_date": None
                }

        # Mapear información al DataFrame original
        df["competition_name"] = df["competition_id"].map(lambda x: comp_info.get(x, {}).get("competition_name"))
        df["start_date"] = pd.to_datetime(
            df["competition_id"].map(lambda x: comp_info.get(x, {}).get("start_date")),
            errors='coerce'
        )

        # Filtrar por año
        df_filtered = df[df["start_date"].dt.year.isin([2024, 2025])].copy()

        # Convertir centisegundos → segundos
        df_filtered["single_seconds"] = df_filtered["best"].apply(lambda x: round(x / 100, 2) if x > 0 else None)
        df_filtered["average_seconds"] = df_filtered["average"].apply(lambda x: round(x / 100, 2) if x > 0 else None)

        # Ordenar por fecha
        df_filtered = df_filtered.sort_values("start_date")

        # Guardar CSV
        df_filtered.to_csv(f"wca_api_resultados_{wca_id}_2024_2025.csv", index=False)
        print(f"✅ CSV guardado con {len(df_filtered)} resultados.")
    else:
        print("⚠️ El competidor no tiene resultados o 'competition_id' está ausente.")
else:
    print(f"❌ Error al conectar con la API: {response.status_code}")
