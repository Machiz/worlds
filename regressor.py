import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Cargar el modelo Random Forest guardado ---
rf_model_filename = 'random_forest_reg.joblib'
loaded_rf_model = joblib.load(rf_model_filename) #
print(f"Modelo RandomForest '{rf_model_filename}' cargado exitosamente.")

# --- 2. Definir tu lista específica de nombres válidos ---
# Solo necesitamos los nombres para iterar y para el LabelEncoder.
lista_nombres_validos = [
    "Bofan Zhang",
    "Luke Garrett",
    "Matty Hiroto Inaba",
    "Max Park",
    "Ruihang Xu",
    "Teodor Zajder",
    "Tymon Kolasiński",
    "Xuanyi Geng",
    "Yiheng Wang"
]

# --- 3. Recrear y ajustar el LabelEncoder ---
runtime_label_encoder = LabelEncoder() #
runtime_label_encoder.fit(lista_nombres_validos) #

# --- 4. Lista para almacenar los resultados finales ---
resultados_finales = []

print("\n--- Modo de Predicción por Lista de Personas ---")
print("Se te pedirá que ingreses la posición y el tiempo promedio para cada persona.")

# --- 5. Iterar sobre cada nombre en la lista ---
for person_name_input_str in lista_nombres_validos:
    print(f"\n--- Procesando a: {person_name_input_str} ---")

    # Solicitar el tiempo promedio para esa persona
    while True:
        try:
            avg_seconds_input_str = input(f"Ingrese el tiempo promedio en segundos para '{person_name_input_str}': ").strip()
            new_avg_seconds = float(avg_seconds_input_str)
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número para los segundos.")

    # Solicitar la posición para esa persona
    while True:
        try:
            pos_input_str = input(f"Ingrese la posición (número entero, ej. 1, 2, 3...) para '{person_name_input_str}': ").strip()
            new_pos = int(pos_input_str)
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número entero para la posición.")

    # --- 6. Preprocesar el nombre de entrada (codificar) ---
    encoded_name = runtime_label_encoder.transform([person_name_input_str])[0] #
    print(f"'{person_name_input_str}' codificado a: {encoded_name}")

    # --- 7. Preparar los datos para la predicción (average_seconds, name y pos) ---
    new_data_for_prediction = pd.DataFrame({
        'average_seconds': [new_avg_seconds],
        'name': [encoded_name],
        'pos': [new_pos] # Incluimos la posición ingresada por el usuario
    })

    print(f"Datos preprocesados para predicción: {new_data_for_prediction.iloc[0].to_dict()}")

    # --- 8. Hacer predicciones ---
    predictions = loaded_rf_model.predict(new_data_for_prediction) #

    if hasattr(loaded_rf_model, 'predict_proba'):
        probabilities = loaded_rf_model.predict_proba(new_data_for_prediction) #
        print(f"Probabilidad de NO EN PODIO: {probabilities[0][0]:.3f}, Probabilidad de EN PODIO: {probabilities[0][1]:.3f}")

    class_mapping = {0: 'NO EN PODIO', 1: 'EN PODIO'}
    predicted_outcome = class_mapping.get(predictions[0], "Desconocido")

    print(f"La predicción para '{person_name_input_str}' es: **{predicted_outcome}** (valor numérico: {predictions[0]})")

    # --- Almacenar el resultado ---
    resultados_finales.append({
        'nombre': person_name_input_str,
        'prediccion_numerica': predictions[0],
        'prediccion_texto': predicted_outcome,
        'average_seconds_input': new_avg_seconds,
        'pos_input': new_pos # Almacenamos la posición ingresada por el usuario
    })

print("\n--- ¡Proceso de predicción completado para todas las personas! ---")

# --- 9. Imprimir la lista de resultados finales ordenada como un podio ---
print("\n--- Posible Podio Final (Basado en Predicciones) ---")

if resultados_finales:
    resultados_finales_ordenados = sorted(
        resultados_finales,
        key=lambda x: (x['prediccion_numerica'], x['average_seconds_input']),
        reverse=True
    )

    df_resultados = pd.DataFrame(resultados_finales_ordenados)

    print("\n--- Candidatos a EN PODIO ---")
    podio_candidatos = df_resultados[df_resultados['prediccion_numerica'] == 1].head(3)
    if not podio_candidatos.empty:
        for i, row in podio_candidatos.iterrows():
            print(f"  Puesto {i+1}: {row['nombre']} (Tiempo: {row['average_seconds_input']:.2f}s, Posición ingresada: {row['pos_input']})")
    else:
        print("No hay personas predichas para estar EN PODIO.")

    print("\n--- Otros Resultados (Ordenados por Predicción y Tiempo) ---")
    if not df_resultados.empty:
        nombres_en_podio = podio_candidatos['nombre'].tolist()
        
        for i, row in df_resultados.iterrows():
            if row['prediccion_numerica'] == 0 or (row['prediccion_numerica'] == 1 and row['nombre'] not in nombres_en_podio):
                print(f"  {row['nombre']} (Tiempo: {row['average_seconds_input']:.2f}s, Predicción: {row['prediccion_texto']}, Posición ingresada: {row['pos_input']})")
    else:
        print("No hay resultados para mostrar.")
else:
    print("No se realizaron predicciones.")