# Importar bibliotecas necesarias
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Abrir explorador de archivos para seleccionar el CSV
def seleccionar_archivo():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.attributes('-topmost', True)  # Mantener el diálogo sobre otras ventanas
    file_path = askopenfilename(filetypes=[("Archivos CSV", "*.csv")], title="Selecciona un archivo CSV")
    return file_path

# Solicitar el archivo CSV
print("Selecciona el archivo CSV desde el explorador.")
file_path = seleccionar_archivo()

# Verificar si se seleccionó un archivo
if not file_path:
    print("No seleccionaste ningún archivo. El programa finalizará.")
    exit()

# Leer los datos desde el archivo CSV
try:
    df = pd.read_csv(file_path)
    print("Archivo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Vista inicial de los datos
print("Primeras filas de los datos:")
print(df.head())

# Comprobar columnas obligatorias
required_columns = ['idProducto', 'nombreProducto', 'cantidad', 'total', 'fecha1', 'fecha2']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"El archivo CSV no contiene las columnas necesarias: {missing_columns}")
    exit()

# Validar y convertir las columnas 'fecha1' y 'fecha2' a un formato datetime
try:
    df['fecha1'] = pd.to_datetime(df['fecha1'], format="%Y-%m-%d")  # Asegurar formato de año, mes y día
    df['fecha2'] = pd.to_datetime(df['fecha2'], format="%H:%M:%S").dt.time  # Asegurar formato de hora, minuto y segundo
except Exception as e:
    print(f"Error al procesar las columnas de fecha: {e}")
    exit()

# Combinar 'fecha1' y 'fecha2' en una sola columna 'fecha_hora'
df['fecha_hora'] = df.apply(lambda row: pd.Timestamp.combine(row['fecha1'], row['fecha2']), axis=1)

# Extraer características adicionales
# Extraer mes, día y día de la semana para análisis y predicciones
df['mes'] = df['fecha1'].dt.month
df['dia'] = df['fecha1'].dt.day
df['dia_semana'] = df['fecha1'].dt.weekday

# Asegurar que idProducto sea numérico
df['idProducto'] = pd.to_numeric(df['idProducto'], errors='coerce')

# Eliminar filas con valores inválidos en idProducto
df = df.dropna(subset=['idProducto'])

# Variables independientes (X) y dependiente (y)
X = df[['idProducto', 'total', 'mes', 'dia', 'dia_semana']]
y = df['cantidad']

# División de datos y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo y entrenar
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
# Calcular métricas de error y ajuste del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluación del Modelo ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Predicciones futuras: Generar un dataset para un rango de fechas y horas específicas
print("\n--- Predicción de Demanda Futura ---")
fechas_futuras = pd.date_range(start='2025-01-12', periods=24, freq='H')  # Generar horas del día 12 de enero de 2025
predicciones_futuras = []

# Iterar sobre cada fecha y producto para hacer predicciones
for fecha in fechas_futuras:
    hora = fecha.hour  # Extraer hora
    dia_semana = fecha.weekday()  # Extraer día de la semana
    mes = fecha.month  # Extraer mes
    dia = fecha.day  # Extraer día del mes
    for id_producto in df['idProducto'].unique():
        # Obtener nombre del producto y calcular el promedio de ventas (total)
        nombre_producto = df[df['idProducto'] == id_producto]['nombreProducto'].iloc[0]
        total_promedio = df[df['idProducto'] == id_producto]['total'].mean()

        # Crear un conjunto de datos con características para predicción
        datos = pd.DataFrame({
            'idProducto': [id_producto],
            'total': [total_promedio],
            'mes': [mes],
            'dia': [dia],
            'dia_semana': [dia_semana]
        })

        # Hacer la predicción para el producto en la fecha específica
        demanda_predicha = model.predict(datos)[0]
        predicciones_futuras.append({
            'fecha_hora': fecha,
            'idProducto': id_producto,
            'nombreProducto': nombre_producto,
            'demanda_predicha': demanda_predicha
        })

# Convertir las predicciones futuras a un DataFrame
predicciones_df = pd.DataFrame(predicciones_futuras)
print(predicciones_df.head())  # Mostrar las primeras filas de las predicciones

# Guardar las predicciones futuras en un archivo CSV
output_file = "predicciones_demanda_futura.csv"
predicciones_df.to_csv(output_file, index=False)
print(f"Predicciones futuras guardadas en el archivo: {output_file}")
