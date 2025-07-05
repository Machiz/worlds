import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Cargar el modelo Random Forest guardado ---
rf_model_filename = 'random_forest.joblib'
try:
    loaded_rf_model = joblib.load(rf_model_filename)
    print(f"Modelo RandomForest '{rf_model_filename}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo del modelo '{rf_model_filename}' no se encontró. Asegúrate de que esté en la misma ubicación o proporciona la ruta completa.")
    exit()

# --- 2. Definir tu lista específica de nombres válidos ---
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
runtime_label_encoder = LabelEncoder()
runtime_label_encoder.fit(lista_nombres_validos)

# --- 4. Lista para almacenar los resultados finales ---
resultados_finales = []

print("\n--- Modo de Predicción por Lista de Personas ---")
print("Se te pedirá que ingreses el tiempo promedio para cada persona en la lista.")

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

    # --- 6. Preprocesar el nombre de entrada (codificar) ---
    try:
        encoded_name = runtime_label_encoder.transform([person_name_input_str])[0]
        print(f"'{person_name_input_str}' codificado a: {encoded_name}")
    except ValueError as e:
        print(f"Error: El nombre '{person_name_input_str}' no pudo ser codificado por el LabelEncoder. {e}")
        print("Asegúrate de que 'lista_nombres_validos' contenga todos los nombres que el modelo vio durante el entrenamiento.")
        continue

    # --- 7. Preparar los datos para la predicción ---
    new_data_for_prediction = pd.DataFrame({
        'average_seconds': [new_avg_seconds],
        'name': [encoded_name],
        #'en_podio_prob':[ 0.519231, 0.939024, 0.882353, 0.986301, 0.645161, 0.807692, 1.000000, 0.983051, 0.977099]
    })

    print(f"Datos preprocesados para predicción: {new_data_for_prediction.iloc[0].to_dict()}")

    # --- 8. Hacer predicciones ---
    try:
        predictions = loaded_rf_model.predict(new_data_for_prediction)

        if hasattr(loaded_rf_model, 'predict_proba'):
            probabilities = loaded_rf_model.predict_proba(new_data_for_prediction)
            print("Probabilidades de las clases:")
            print(np.round(probabilities, 3))

        class_mapping = {0: 'NO EN PODIO', 1: 'EN PODIO'}
        predicted_outcome = class_mapping.get(predictions[0], "Desconocido")

        print(f"La predicción para '{person_name_input_str}' es: **{predicted_outcome}** (valor numérico: {predictions[0]})")

        # --- Almacenar el resultado ---
        resultados_finales.append({
            'nombre': person_name_input_str,
            'prediccion_numerica': predictions[0], # 1 para EN PODIO, 0 para NO EN PODIO
            'prediccion_texto': predicted_outcome,
            'average_seconds_input': new_avg_seconds
        })

    except Exception as e:
        print(f"\nError al realizar la predicción para '{person_name_input_str}': {e}")
        print("Verifica que las columnas y su orden en 'new_data_for_prediction' coincidan con 'X_train'.")

print("\n--- ¡Proceso de predicción completado para todas las personas! ---")

# --- 9. Imprimir la lista de posiciones finales ordenada como un podio ---
print("\n--- Posible Podio Final (Basado en Predicciones) ---")

if resultados_finales:
    # Ordenar los resultados:
    # 1. Primero por 'prediccion_numerica' en orden descendente (1 antes que 0)
    # 2. Luego por 'average_seconds_input' en orden ascendente (tiempo menor es mejor)
    resultados_finales_ordenados = sorted(
        resultados_finales,
        key=lambda x: (x['prediccion_numerica'], -x['average_seconds_input']), # Nota el signo negativo aquí para segundos
        reverse=True # Con reverse=True, 1 va antes que 0.
    )

    # Convertir a un DataFrame de Pandas para una mejor visualización y fácil acceso a top-N
    df_resultados = pd.DataFrame(resultados_finales_ordenados)

    print("\n--- Candidatos a EN PODIO ---")
    podio_candidatos = df_resultados[df_resultados['prediccion_numerica'] == 1].head(3) # Top 3 del podio
    if not podio_candidatos.empty:
        # Aquí puedes decidir si quieres mostrar solo 3 o todos los que son "En Podio"
        # Mostraremos el top 3 para el podio real
        for i, row in podio_candidatos.iterrows():
            print(f"  Puesto {i+1}: {row['nombre']} (Tiempo: {row['average_seconds_input']:.2f}s)")
    else:
        print("No hay personas predichas para estar EN PODIO.")

    print("\n--- Otros Resultados (Ordenados por Predicción y Tiempo) ---")
    # Imprimimos el resto, o todos si no hubo suficientes para un podio de 3
    # Excluimos los que ya mostramos en el podio explícito
    if not df_resultados.empty:
        for i, row in df_resultados.iterrows():
            # Si no está en el top 3 del podio
            if row['prediccion_numerica'] == 0 or (row['prediccion_numerica'] == 1 and i >= 3):
                print(f"  {row['nombre']} , Tiempo: {row['average_seconds_input']:.2f}s)")
    else:
        print("No hay resultados para mostrar.")


else:
    print("No se realizaron predicciones.")