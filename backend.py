"""
Módulo Backend: procesamiento de datos y componente predictivo del
Sistema de Información Inteligente "Entrenador IA FCDIA".

Contiene la lógica de negocio (carga del CSV, limpieza, feature engineering,
agregaciones e indicadores). No depende de Streamlit, por lo que puede probarse
o reutilizarse de forma independiente al frontend (app.py).
"""

import pandas as pd                                                                 # Manejo y análisis de datos tabulares


# ---------------------------------------------------------
# CONSTANTES DE CONFIGURACIÓN DE DATOS
# ---------------------------------------------------------
RUTA_CSV = "actividades_strava.csv"                                                 # Ruta por defecto del archivo de actividades

COLUMNAS_DESEADAS = [                                                               # Columnas de origen que intentamos conservar
    "Fecha de la actividad",
    "Tipo de actividad",
    "Distancia",
    "Tiempo en movimiento",
    "Desnivel positivo",
    "Ritmo cardiaco promedio",
    "Velocidad promedio",
]

COLUMNAS_NUMERICAS = [                                                              # Columnas que deben convertirse a número
    "Distancia",
    "Tiempo en movimiento",
    "Desnivel positivo",
    "Ritmo cardiaco promedio",
    "Velocidad promedio",
]

MESES_ES_EN = {                                                                     # Diccionario de meses español -> inglés
    "ene": "jan", "feb": "feb", "mar": "mar", "abr": "apr",
    "may": "may", "jun": "jun", "jul": "jul", "ago": "aug",
    "sep": "sep", "oct": "oct", "nov": "nov", "dic": "dec",
}


def cargar_y_procesar_datos(ruta_csv=RUTA_CSV):
    """
    Carga el CSV de Strava, filtra columnas, limpia números, traduce fechas del
    español al inglés, imputa nulos y calcula variables derivadas.

    Devuelve un DataFrame limpio y ordenado temporalmente, o None si el archivo
    no existe.
    """
    try:
        df = pd.read_csv(ruta_csv)                                                  # Lee el archivo CSV de actividades
    except FileNotFoundError:
        return None                                                                # Señaliza al frontend que no hay archivo

    columnas_presentes = [c for c in COLUMNAS_DESEADAS if c in df.columns]          # Detecta qué columnas existen realmente
    df = df[columnas_presentes].copy()                                             # Se queda solo con las columnas útiles

    for col in COLUMNAS_NUMERICAS:                                                  # Recorre las columnas numéricas objetivo
        if col in df.columns:                                                      # Procesa solo si la columna está presente
            serie_texto = df[col].astype(str).str.replace(",", ".", regex=False)   # Cambia coma decimal por punto
            df[col] = pd.to_numeric(serie_texto, errors="coerce")                  # Convierte a número (NaN si falla)

    if "Ritmo cardiaco promedio" in df.columns:                                    # Solo imputa si existe la columna de FC
        media_pulsos = df["Ritmo cardiaco promedio"].mean()                        # Calcula la media global de pulsaciones
        df["Ritmo cardiaco promedio"] = df["Ritmo cardiaco promedio"].fillna(media_pulsos)  # Rellena los días sin reloj

    if "Tiempo en movimiento" in df.columns:                                       # Requiere el tiempo para calcular minutos
        df["Minutos"] = df["Tiempo en movimiento"] / 60                            # Convierte segundos a minutos

    if {"Minutos", "Ritmo cardiaco promedio"}.issubset(df.columns):                # Necesita minutos y FC para el TRIMP
        df["Carga Estimada"] = (df["Minutos"] * df["Ritmo cardiaco promedio"]) / 100  # TRIMP simplificado (impulso de entreno)

    if "Distancia" in df.columns:                                                  # Solo si tenemos distancia en metros
        df["Distancia_km"] = df["Distancia"] / 1000                                # Convierte metros a kilómetros

    if {"Minutos", "Distancia_km"}.issubset(df.columns):                           # Ritmo requiere minutos y km
        df["Ritmo (min/km)"] = (df["Minutos"] / df["Distancia_km"]).where(df["Distancia_km"] > 0)  # Evita dividir por cero

    if "Fecha de la actividad" in df.columns:                                      # Procesa las fechas solo si existen
        fechas_str = df["Fecha de la actividad"].astype(str).str.lower()           # Pasa la fecha a texto en minúsculas
        for esp, eng in MESES_ES_EN.items():                                       # Recorre el diccionario de meses
            fechas_str = fechas_str.str.replace(esp, eng, regex=False)             # Traduce cada abreviatura de mes
        df["Fecha"] = pd.to_datetime(fechas_str, errors="coerce")                  # Parsea la fecha a datetime real
        df = df.dropna(subset=["Fecha"])                                           # Descarta filas sin fecha válida
        df = df.sort_values("Fecha").reset_index(drop=True)                        # Ordena de la más antigua a la más reciente
        df["Semana"] = df["Fecha"].dt.to_period("W").dt.start_time                 # Etiqueta la semana (lunes) de cada actividad

    return df                                                                      # Devuelve el DataFrame ya procesado


def predecir_riesgo_fatiga(df):
    """
    Estima el riesgo de lesión mediante el ratio de carga aguda vs crónica
    (últimas 7 actividades frente a las últimas 28). Devuelve una tupla
    (estado_texto, ratio_numerico).
    """
    if "Carga Estimada" not in df.columns or len(df) < 10:                         # Requiere carga y datos suficientes
        return "Insuficientes datos", 1.0                                          # Estado neutro si faltan datos

    carga_aguda = df["Carga Estimada"].tail(7).mean()                              # Carga reciente (fatiga)
    carga_cronica = df["Carga Estimada"].tail(28).mean()                           # Carga histórica (fitness)
    ratio_fatiga = carga_aguda / carga_cronica if carga_cronica > 0 else 1.0       # Ratio agudo:crónico

    if ratio_fatiga > 1.5:                                                         # Umbral superior de sobrecarga
        return "ALTO RIESGO DE LESIÓN", ratio_fatiga                               # Alerta alta
    elif ratio_fatiga < 0.8:                                                       # Umbral inferior de baja carga
        return "SUB-ENTRENADO (Falta carga)", ratio_fatiga                         # Alerta baja
    else:
        return "ESTADO ÓPTIMO", ratio_fatiga                                       # Zona segura


def calcular_kpis(df):
    """
    Calcula los indicadores clave (KPIs) del panel a partir del DataFrame ya
    procesado. Devuelve un diccionario listo para mostrar con st.metric.
    """
    kpis = {}                                                                      # Contenedor de indicadores
    kpis["num_actividades"] = len(df)                                              # Número total de actividades

    if "Distancia_km" in df.columns:                                              # KPI de distancia acumulada
        kpis["distancia_total_km"] = df["Distancia_km"].sum()                      # Suma total en kilómetros
    if "Ritmo cardiaco promedio" in df.columns:                                    # KPI de frecuencia cardiaca
        kpis["fc_promedio"] = df["Ritmo cardiaco promedio"].mean()                 # FC media global
    if "Ritmo (min/km)" in df.columns:                                            # KPI de ritmo medio
        kpis["ritmo_promedio"] = df["Ritmo (min/km)"].mean()                       # Ritmo medio en min/km
    if "Carga Estimada" in df.columns:                                            # KPI de carga media
        kpis["carga_promedio"] = df["Carga Estimada"].mean()                       # Carga media por sesión
    if "Desnivel positivo" in df.columns:                                         # KPI de desnivel
        kpis["desnivel_total"] = df["Desnivel positivo"].sum()                     # Desnivel positivo acumulado

    return kpis                                                                     # Devuelve el diccionario de KPIs


def serie_carga_acwr(df, ventana_aguda=7, ventana_cronica=28):
    """
    Construye la serie temporal del ratio agudo:crónico (ACWR) usando medias
    móviles de la 'Carga Estimada'. Devuelve una copia del DataFrame con las
    columnas 'Carga Aguda', 'Carga Cronica' y 'ACWR'.
    """
    if "Carga Estimada" not in df.columns:                                         # Sin carga no hay ACWR posible
        return df.copy()                                                           # Devuelve copia sin cambios

    resultado = df.copy()                                                          # Trabaja sobre una copia segura
    resultado["Carga Aguda"] = resultado["Carga Estimada"].rolling(ventana_aguda, min_periods=1).mean()      # Media móvil corta
    resultado["Carga Cronica"] = resultado["Carga Estimada"].rolling(ventana_cronica, min_periods=1).mean()  # Media móvil larga
    resultado["ACWR"] = resultado["Carga Aguda"] / resultado["Carga Cronica"]      # Ratio agudo:crónico por punto
    return resultado                                                               # Devuelve la serie enriquecida


def carga_por_semana(df):
    """
    Agrupa la carga y la distancia por semana para visualizar tendencias
    semanales. Devuelve un DataFrame agregado (o vacío si faltan columnas).
    """
    if "Semana" not in df.columns:                                                 # Requiere la etiqueta de semana
        return pd.DataFrame()                                                      # Devuelve vacío si no aplica

    agregaciones = {}                                                              # Diccionario dinámico de agregaciones
    if "Carga Estimada" in df.columns:                                            # Agrega carga si existe
        agregaciones["Carga Estimada"] = "sum"                                     # Suma de carga semanal
    if "Distancia_km" in df.columns:                                              # Agrega distancia si existe
        agregaciones["Distancia_km"] = "sum"                                       # Suma de km semanales

    if not agregaciones:                                                           # Si no hay nada que agregar
        return pd.DataFrame()                                                      # Devuelve vacío

    semanal = df.groupby("Semana").agg(agregaciones).reset_index()                 # Agrupa por semana y agrega
    return semanal                                                                 # Devuelve el resumen semanal


def resumen_por_tipo(df):
    """
    Resume el número de actividades y la distancia total por tipo de actividad.
    Devuelve un DataFrame ordenado (o vacío si no existe la columna de tipo).
    """
    if "Tipo de actividad" not in df.columns:                                      # Requiere la columna de tipo
        return pd.DataFrame()                                                      # Devuelve vacío si no aplica

    columnas_agg = {}                                                              # Diccionario dinámico de agregaciones
    if "Fecha" in df.columns:                                                      # Cuenta actividades usando la fecha
        columnas_agg["Fecha"] = "count"                                            # Conteo de actividades por tipo
    if "Distancia_km" in df.columns:                                              # Suma distancia por tipo si existe
        columnas_agg["Distancia_km"] = "sum"                                       # Distancia total por tipo

    if not columnas_agg:                                                           # Si no hay nada que agregar
        return pd.DataFrame()                                                      # Devuelve vacío

    resumen = df.groupby("Tipo de actividad").agg(columnas_agg).reset_index()      # Agrupa por tipo de actividad
    resumen = resumen.rename(columns={"Fecha": "Actividades"})                     # Renombra el conteo a 'Actividades'
    orden = "Actividades" if "Actividades" in resumen.columns else resumen.columns[-1]  # Elige criterio de orden
    resumen = resumen.sort_values(orden, ascending=False).reset_index(drop=True)   # Ordena de mayor a menor
    return resumen                                                                 # Devuelve el resumen por tipo