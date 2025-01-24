# Importar bibliotecas necesarias
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Función para seleccionar el archivo CSV
def seleccionar_archivo():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = askopenfilename(filetypes=[("Archivos CSV", "*.csv")], title="Selecciona un archivo CSV")
    return file_path

# Solicitar archivo CSV
print("Selecciona el archivo CSV desde el explorador.")
file_path = seleccionar_archivo()

if not file_path:
    print("No seleccionaste ningún archivo. El programa finalizará.")
    exit()

# Leer los datos
try:
    df = pd.read_csv(file_path)
    print("Archivo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Validación de columnas requeridas
required_columns = ['idProducto', 'nombreProducto', 'cantidad', 'total', 'fecha1']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"El archivo CSV no contiene las columnas necesarias: {missing_columns}")
    exit()

# Convertir fechas
try:
    df['fecha1'] = pd.to_datetime(df['fecha1'], format="%Y-%m-%d")
except Exception as e:
    print(f"Error al procesar las fechas: {e}")
    exit()

# Extraer características de la fecha
df['mes'] = df['fecha1'].dt.month
df['dia'] = df['fecha1'].dt.day
df['dia_semana'] = df['fecha1'].dt.weekday
df['is_weekend'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)  # 1 si es fin de semana

# Asegurar que idProducto sea numérico
df['idProducto'] = pd.to_numeric(df['idProducto'], errors='coerce')
df = df.dropna(subset=['idProducto'])  # Eliminar filas con valores nulos en idProducto

# Agrupar por día y producto para obtener la cantidad total vendida por día
df_grouped = df.groupby(['fecha1', 'idProducto', 'nombreProducto']).agg({
    'cantidad': 'sum',
    'total': 'sum'
}).reset_index()

# Generar características adicionales
df_grouped['mes'] = df_grouped['fecha1'].dt.month
df_grouped['dia'] = df_grouped['fecha1'].dt.day
df_grouped['dia_semana'] = df_grouped['fecha1'].dt.weekday
df_grouped['is_weekend'] = df_grouped['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

# Definir variables X e y
X = df_grouped[['idProducto', 'total', 'mes', 'dia', 'dia_semana', 'is_weekend']]
y = df_grouped['cantidad']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_
print("\nMejores hiperparámetros encontrados:", grid_search.best_params_)

# Predicciones
y_pred = best_model.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cálculo de porcentaje de precisión basado en el error absoluto
mean_actual = np.mean(y_test)
precision = (1 - (mae / mean_actual)) * 100 if mean_actual != 0 else 0

print("\n--- Evaluación del Modelo ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Precisión del modelo: {precision:.2f}%")

# Predicción de demanda futura por días
print("\n--- Predicción de Demanda Futura ---")
ultima_fecha = df_grouped['fecha1'].max()
fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=3, freq='D')

predicciones_futuras = []
for fecha in fechas_futuras:
    dia_semana = fecha.weekday()
    mes = fecha.month
    dia = fecha.day
    is_weekend = 1 if dia_semana >= 5 else 0
    for id_producto in df_grouped['idProducto'].unique():
        nombre_producto = df_grouped[df_grouped['idProducto'] == id_producto]['nombreProducto'].iloc[0]
        total_promedio = df_grouped[df_grouped['idProducto'] == id_producto]['total'].mean()

        datos = pd.DataFrame({
            'idProducto': [id_producto],
            'total': [total_promedio],
            'mes': [mes],
            'dia': [dia],
            'dia_semana': [dia_semana],
            'is_weekend': [is_weekend]
        })

        demanda_predicha = best_model.predict(datos)[0]
        predicciones_futuras.append({
            'fecha': fecha.date(),
            'idProducto': id_producto,
            'nombreProducto': nombre_producto,
            'demanda_predicha': round(demanda_predicha, 2)
        })

# Convertir predicciones a DataFrame
predicciones_df = pd.DataFrame(predicciones_futuras)
print(predicciones_df.head())

# Guardar en CSV
output_file = "predicciones_demanda_futura_mejorado.csv"
predicciones_df.to_csv(output_file, index=False)
print(f"Predicciones futuras guardadas en el archivo: {output_file}")
