import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re # Importamos el módulo re para expresiones regulares

# --- 1. Cargar el modelo y el LabelEncoder ---
# ASEGÚRATE de que este modelo sea un RandomForestRegressor y fue entrenado con 'pos' como objetivo.
filename_model = 'random_forest.joblib' # ¡Cambiado a nombre de regresor!
filename_encoder = 'label_encoder.joblib' # Asegúrate de que este archivo exista

try:
    loaded_model = joblib.load(filename_model)
    print(f"Modelo '{filename_model}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del modelo '{filename_model}'. "
          "Asegúrate de que el modelo RandomForestRegressor haya sido entrenado y guardado correctamente.")
    exit()

label_encoder = None
try:
    label_encoder = joblib.load(filename_encoder)
    print(f"LabelEncoder '{filename_encoder}' cargado exitosamente.")
    print(f"El LabelEncoder conoce {len(label_encoder.classes_)} nombres.")
    print(f"Clases conocidas por LabelEncoder (ej. 3 primeras): {label_encoder.classes_[:3]}...")
except FileNotFoundError:
    print(f"Advertencia: No se encontró el LabelEncoder '{filename_encoder}'.")
    print("Intentando recrearlo con una lista fija de nombres. ESTO ES SÓLO PARA DEPURACIÓN Y NO ES RECOMENDADO PARA PRODUCCIÓN.")
    print("La mejor práctica es cargar el LabelEncoder que fue ajustado con todos los nombres de tu data de entrenamiento original.")
    
    # Esta lista DEBE contener exactamente todos los nombres únicos que el encoder vio durante el entrenamiento
    # Y DEBEN SER LOS NOMBRES NORMALIZADOS (sin el texto entre paréntesis, etc.)
    all_known_names_from_training = [
        'Bofan Zhang', 'Luke Garrett', 'Matty Hiroto Inaba', 'Max Park',
        'Ruihang Xu', 'Teodor Zajder', 'Tymon Kolasiński', 'Xuanyi Geng',
        'Yiheng Wang'
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_known_names_from_training)
    print("LabelEncoder recreado con una lista fija. Asegúrate de que esta lista sea exhaustiva y con nombres normalizados.")
    print(f"El LabelEncoder recreado conoce {len(label_encoder.classes_)} nombres.")
    print(f"Clases conocidas por LabelEncoder (ej. 3 primeras): {label_encoder.classes_[:3]}...")

if label_encoder is None:
    print("Error crítico: No se pudo cargar ni recrear el LabelEncoder. Saliendo.")
    exit()

original_feature_names = ['average_seconds', 'name_encoded'] # Actualizado para reflejar la característica codificada
print(f"El modelo fue entrenado con las características: {original_feature_names}")


# --- 2. Preparar los nuevos inputs ---
new_data_raw = pd.DataFrame({
    'name': [ # Usar 'name' directamente para que coincida con la lógica interna
        "Bofan Zhang (comp)", # Ejemplo de texto adicional
        "Luke Garrett",
        "Matty Hiroto Inaba",
        "Max Park Jr.", # Otro ejemplo, si el entrenamiento fue sin "Jr."
        "Ruihang Xu",
        "Teodor Zajder",
        "Tymon Kolasiński",
        "Xuanyi Geng (sub 18)", # Otro ejemplo con paréntesis
        "Yiheng Wang"
    ],
    'promedio_ultimos': [5.68, 6.25, 5.83, 5.64, 5.38, 5.82, 5.20, 4.80, 4.51]
})

new_data_processed = new_data_raw.copy()

# Renombrar 'promedio_ultimos' a 'average_seconds' para que coincida con lo que el modelo espera
new_data_processed.rename(columns={'promedio_ultimos': 'average_seconds'}, inplace=True)

try:
    # NORMALIZACIÓN DE NOMBRES CON REGEX (igual que en el entrenamiento)
    new_data_processed['name_normalized'] = new_data_processed['name'].apply(
        lambda x: re.sub(r'\s*\([^)]*\)$', '', str(x)).strip()
    )
    # Si quieres quitar "Jr.", "Sr.", etc., o cualquier caracter no alfanumérico al final:
    # new_data_processed['name_normalized'] = new_data_processed['name_normalized'].apply(
    #     lambda x: re.sub(r'\s*(Jr\.|Sr\.|[IVX]+\.?|[,-]\s*.*)$', '', str(x)).strip()
    # )
    
    new_data_processed['name_encoded'] = label_encoder.transform(new_data_processed['name_normalized'])
    
    # Decodificación de nombres para verificación (opcional, pero útil)
    new_data_processed['name_decoded_check'] = label_encoder.inverse_transform(new_data_processed['name_encoded'])
    print("\n--- Verificación de Codificación/Decodificación de Nombres ---")
    print(new_data_processed[['name', 'name_normalized', 'name_encoded', 'name_decoded_check']].head())

except ValueError as e:
    print(f"\nError al codificar nombres: {e}")
    unseen_names = [name for name in new_data_processed['name_normalized'].unique() if name not in label_encoder.classes_]
    print(f"Los siguientes nombres normalizados no fueron vistos durante el entrenamiento del LabelEncoder: {unseen_names}")
    print("Asegúrate de que todos los nombres en tus nuevos inputs (después de la limpieza con regex) existieran en los datos de entrenamiento del LabelEncoder.")
    exit()

# Seleccionar solo las columnas que el modelo espera y en el orden correcto
new_data_for_prediction = new_data_processed[['average_seconds', 'name_encoded']]

print("\n--- Datos de Entrada para Predicción (Preprocesados) ---")
print(new_data_for_prediction)

# --- 3. Realizar Predicciones de POSICIÓN (Regresión) ---
# ¡USAMOS .predict() para obtener el valor numérico de la posición!
predicted_positions = loaded_model.predict(new_data_for_prediction)
print("\n--- Posiciones Predichas para cada Competidor (valores raw) ---")
print(predicted_positions)

# Añadir las posiciones predichas al DataFrame original de los inputs
new_data_raw['predicted_pos'] = predicted_positions

# --- 4. Ordenar la lista completa de posiciones ---
# Ordenar por la posición predicha de MENOR a MAYOR (posición más baja es mejor)
df_ranking_prediction_results = new_data_raw.sort_values(by='predicted_pos', ascending=True).reset_index(drop=True)

print("\n--- Lista Completa de Posiciones Predichas ---")
# Imprimimos la lista completa, incluyendo un índice de ranking
for i, row in enumerate(df_ranking_prediction_results.itertuples(), 1):
    original_name = row.name
    predicted_pos = row.predicted_pos
    # Se usa .2f porque la regresión puede dar valores flotantes
    print(f"{i}° Lugar: {original_name} (Posición Predicha: {predicted_pos:.2f})")


# ==============================================================================
# --- SECCIÓN DE GRÁFICOS (Adaptados para Regresión de Posición) ---
# ==============================================================================
# Los gráficos ya estaban adaptados para la regresión en la versión anterior.
# Solo asegúrate de que 'name' y 'promedio_ultimos' existan en el DataFrame usado para el plot.

# Gráfico 1: Posición Predicha vs. Tiempo Promedio
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_ranking_prediction_results,
    x='promedio_ultimos', # Usamos el nombre original para el gráfico
    y='predicted_pos',
    hue='name', # Nombre original para la leyenda
    size='predicted_pos', # El tamaño podría ser inversamente proporcional para mejor visualización
    sizes=(400, 50), # Más grande para posiciones bajas (mejores)
    alpha=0.8
)

plt.title('Posición Predicha vs. Tiempo Promedio de los Competidores')
plt.xlabel('Tiempo Promedio (segundos)')
plt.ylabel('Posición Predicha')
plt.grid(True, linestyle='--', alpha=0.7)

# Anotar los 3 primeros lugares en el gráfico de dispersión
top_3_chart = df_ranking_prediction_results.head(3)
for i, row in top_3_chart.iterrows():
    plt.annotate(
        f"{row['name']} ({row['predicted_pos']:.2f})", # Mostrar nombre y posición predicha
        (row['promedio_ultimos'], row['predicted_pos']),
        textcoords="offset points",
        xytext=(5,-10),
        ha='left',
        fontsize=9,
        color='red'
    )

plt.legend(title='Competidor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
# Gráfico 2: Comparación de Posiciones Predichas para cada Competidor (Ordenado)
plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_ranking_prediction_results,
    x='name', # Nombre original
    y='predicted_pos',
    palette='viridis'
)
plt.title('Posición Predicha para Cada Competidor (Ordenado)')
plt.xlabel('Competidor')
plt.ylabel('Posición Predicha')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""