"""
Módulo Backend del SII "Entrenador IA FCDIA".

Procesa los datos crudos de Strava: normaliza columnas duplicadas, descarta
registros inválidos, calcula la carga de entrenamiento sin falsear los datos
faltantes y deriva el indicador ACWR sobre una serie diaria real.

No depende de Streamlit, por lo que puede probarse de forma independiente.
"""

import numpy as np                                                                 # Operaciones numéricas vectorizadas
import pandas as pd                                                                # Manejo y análisis de datos tabulares

import data_loader                                                                 # Capa de adquisición de datos

# ---------------------------------------------------------
# CONSTANTES DE NEGOCIO
# ---------------------------------------------------------
DEPORTES_DISTANCIA = ["Carrera", "Bicicleta", "Caminata", "Senderismo"]            # Deportes con distancia GPS fiable

RANGOS_RITMO = {                                                                   # Rangos plausibles de ritmo (min/km)
    "Carrera": (2.5, 12.0),                                                        # Descarta paradas y errores de reloj
    "Caminata": (7.0, 25.0),                                                       # Caminata y senderismo son más lentos
    "Senderismo": (7.0, 30.0),                                                     # Terreno de montaña, ritmos altos
    "Bicicleta": (1.0, 10.0),                                                      # Ciclismo es mucho más rápido por km
}

MESES_ES_EN = {                                                                    # Traducción de meses español -> inglés
    "ene": "jan", "abr": "apr", "ago": "aug", "dic": "dec",                        # Solo difieren estas cuatro abreviaturas
}

DIAS_MESOCICLO = 28                                                                # Bloque de carga de 4 semanas


# ---------------------------------------------------------
# LIMPIEZA Y NORMALIZACIÓN
# ---------------------------------------------------------
def _normalizar_distancia(df):
    """
    Resuelve el conflicto de columnas duplicadas del export de Strava.
    El CSV trae 'Distancia' dos veces: la primera en km como texto con coma
    decimal y la segunda (renombrada a 'Distancia.1' por pandas) en metros.
    Se prioriza la versión en metros por ser numérica y fiable.
    """
    if "Distancia.1" in df.columns:                                                # Versión en metros (float limpio)
        metros = pd.to_numeric(df["Distancia.1"], errors="coerce")                 # Fuerza el tipo numérico
        return metros / 1000                                                       # Convierte metros a kilómetros

    texto = df["Distancia"].astype(str).str.replace(",", ".", regex=False)         # Versión en km con coma decimal
    return pd.to_numeric(texto, errors="coerce")                                   # Devuelve los kilómetros ya numéricos


def _parsear_fechas(serie):
    """
    Convierte las fechas de ambas fuentes a datetime sin zona horaria.

    Las dos fuentes usan formatos incompatibles y deben parsearse por separado:
        - CSV exportado: '27 jun 2026, 13:17:16' (día primero, sin zona horaria).
        - API de Strava: '2026-07-12T06:40:01Z' (formato ISO con zona horaria).

    Aplicar 'dayfirst=True' a una fecha ISO invertiría el día y el mes, y mezclar
    fechas con y sin zona horaria impide ordenarlas en una misma serie.
    """
    texto = serie.astype(str).str.lower()                                          # Homogeneiza a minúsculas
    for esp, eng in MESES_ES_EN.items():                                           # Recorre las abreviaturas divergentes
        texto = texto.str.replace(esp, eng, regex=False)                           # Traduce el mes al formato inglés

    es_iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}")                                # Detecta el formato ISO de la API
    fechas = pd.Series(pd.NaT, index=texto.index, dtype="datetime64[ns]")          # Contenedor unificado del resultado

    if es_iso.any():                                                               # Solo si hay registros de la API
        iso = pd.to_datetime(texto[es_iso], errors="coerce", utc=True)             # Parsea ISO respetando la zona horaria
        fechas.loc[es_iso] = iso.dt.tz_localize(None)                              # Descarta la zona para poder comparar

    if (~es_iso).any():                                                            # Solo si hay registros del CSV
        csv = pd.to_datetime(texto[~es_iso], errors="coerce",                      # Parsea el formato español del export
                             dayfirst=True, format="mixed")                        # El día precede al mes en el CSV
        fechas.loc[~es_iso] = csv                                                  # Integra el resultado en la serie

    return fechas                                                                  # Devuelve todas las fechas homogeneizadas


def _filtrar_ritmos_absurdos(df):
    """
    Anula los ritmos fisiológicamente imposibles. Son consecuencia de deportes
    sin distancia (pesas, natación en piscina) y de relojes que se olvidaron
    detener, y son la causa de los picos que distorsionaban las gráficas.
    """
    valido = pd.Series(False, index=df.index)                                      # Máscara inicial: nada es válido

    for deporte, (minimo, maximo) in RANGOS_RITMO.items():                         # Aplica el rango propio de cada deporte
        es_deporte = df["Tipo de actividad"] == deporte                            # Selecciona las filas del deporte
        en_rango = df["Ritmo (min/km)"].between(minimo, maximo)                    # Comprueba el rango plausible
        valido |= (es_deporte & en_rango)                                          # Marca como válidas las que cumplen

    df.loc[~valido, "Ritmo (min/km)"] = np.nan                                     # Anula el resto en lugar de borrar filas
    return df                                                                      # Devuelve el DataFrame corregido


def _calcular_carga(df):
    """
    Calcula la carga de entrenamiento SIN imputar la frecuencia cardíaca.

    El 60 % de las actividades del histórico no tiene FC registrada, por lo que
    rellenarla con la media global falsearía la mayor parte del dataset. En su
    lugar se usa el TRIMP real donde hay pulsómetro y, donde no lo hay, se
    estima la carga a partir de la duración escalada por el factor mediano
    (carga por minuto) observado en ese mismo deporte.
    """
    df["Minutos"] = df["Tiempo en movimiento"] / 60                                # Convierte la duración a minutos
    df["Tiene FC"] = df["Ritmo cardiaco promedio"].notna()                         # Marca la cobertura real del pulsómetro

    df["Carga"] = np.nan                                                           # Inicializa la columna de carga
    con_fc = df["Tiene FC"]                                                        # Selecciona las filas con pulsómetro
    df.loc[con_fc, "Carga"] = (                                                     # TRIMP simplificado (duración x intensidad)
        df.loc[con_fc, "Minutos"] * df.loc[con_fc, "Ritmo cardiaco promedio"] / 100
    )

    for deporte in df["Tipo de actividad"].dropna().unique():                      # Calibra un factor por cada deporte
        es_deporte = df["Tipo de actividad"] == deporte                            # Filas del deporte en cuestión
        referencia = df[es_deporte & con_fc]                                       # Sesiones con FC de ese deporte

        if len(referencia) >= 5 and referencia["Minutos"].sum() > 0:               # Requiere muestra mínima para calibrar
            factor = (referencia["Carga"] / referencia["Minutos"]).median()        # Carga mediana por minuto del deporte
        else:
            factor = 1.4                                                           # Respaldo equivalente a unas 140 ppm

        faltan = es_deporte & ~con_fc                                              # Sesiones sin FC de ese deporte
        df.loc[faltan, "Carga"] = df.loc[faltan, "Minutos"] * factor               # Estima la carga desde la duración

    return df                                                                      # Devuelve el DataFrame con la carga


def cargar_y_procesar_datos():
    """
    Orquesta la carga completa: adquiere las fuentes, normaliza, limpia, calcula
    variables derivadas y ordena temporalmente.

    Devuelve un DataFrame listo para analizar, o None si no hay datos.
    """
    crudo = data_loader.leer_fuentes_crudas()                                      # Une CSV histórico y sincronización API
    if crudo is None or crudo.empty:                                               # Si no existe ninguna fuente de datos
        return None                                                                # Señaliza la ausencia al frontend

    df = pd.DataFrame()                                                            # Contenedor del dataset limpio
    df["Fecha"] = _parsear_fechas(crudo["Fecha de la actividad"])                  # Convierte las fechas a datetime
    df["Tipo de actividad"] = crudo["Tipo de actividad"]                           # Conserva el deporte practicado
    df["Distancia_km"] = _normalizar_distancia(crudo)                              # Resuelve las columnas duplicadas
    df["Tiempo en movimiento"] = pd.to_numeric(crudo["Tiempo en movimiento"], errors="coerce")  # Duración en segundos
    df["Ritmo cardiaco promedio"] = pd.to_numeric(crudo["Ritmo cardiaco promedio"], errors="coerce")  # FC media real
    df["Desnivel positivo"] = pd.to_numeric(crudo["Desnivel positivo"], errors="coerce").fillna(0)  # Desnivel acumulado

    df = df.dropna(subset=["Fecha", "Tiempo en movimiento"])                       # Descarta registros sin fecha ni duración
    df = df[df["Tiempo en movimiento"] > 0]                                        # Elimina actividades de duración nula
    df = df.drop_duplicates(subset=["Fecha"])                                      # Evita duplicados entre CSV y API
    df = df.sort_values("Fecha").reset_index(drop=True)                            # Ordena cronológicamente

    df["Minutos"] = df["Tiempo en movimiento"] / 60                                # Duración en minutos
    ritmo = df["Minutos"] / df["Distancia_km"].replace(0, np.nan)                  # Ritmo bruto evitando dividir por cero
    df["Ritmo (min/km)"] = ritmo.replace([np.inf, -np.inf], np.nan)                # Elimina infinitos residuales
    df = _filtrar_ritmos_absurdos(df)                                              # Anula los ritmos fuera de rango
    df["Velocidad (km/h)"] = df["Distancia_km"] / (df["Minutos"] / 60)             # Velocidad media en km/h
    df.loc[df["Ritmo (min/km)"].isna(), "Velocidad (km/h)"] = np.nan               # Coherencia con el filtro anterior

    df = _calcular_carga(df)                                                       # Calcula la carga sin imputar la FC

    df["Semana"] = df["Fecha"].dt.to_period("W").dt.start_time                     # Etiqueta la semana calendario (lunes)
    df["Año"] = df["Fecha"].dt.year                                                # Año, útil para los filtros
    dias_desde_inicio = (df["Fecha"] - df["Fecha"].min()).dt.days                  # Días transcurridos desde el origen
    df["Mesociclo"] = dias_desde_inicio // DIAS_MESOCICLO                          # Bloque de carga de 4 semanas

    return df                                                                      # Devuelve el dataset procesado


# ---------------------------------------------------------
# INDICADORES Y COMPONENTE DE ANÁLISIS
# ---------------------------------------------------------
def serie_carga_diaria(df, fecha_corte=None):
    """
    Convierte las actividades en una serie diaria continua de carga, rellenando
    con cero los días de descanso. Este paso es imprescindible: el ACWR se
    define sobre ventanas de DÍAS, no sobre las últimas N actividades.
    """
    if df.empty:                                                                   # Sin actividades no hay serie posible
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    fin = fecha_corte if fecha_corte is not None else df["Fecha"].max()            # Permite calcular "a día de hoy"
    diaria = df.set_index("Fecha")["Carga"].resample("D").sum()                    # Suma la carga de cada día natural
    calendario = pd.date_range(df["Fecha"].min().normalize(), pd.Timestamp(fin).normalize(), freq="D")  # Calendario completo
    diaria = diaria.reindex(calendario, fill_value=0)                              # Los días sin entrenar valen cero

    serie = pd.DataFrame({"Fecha": diaria.index, "Carga diaria": diaria.values})   # Estructura la serie resultante
    serie["Carga Aguda"] = serie["Carga diaria"].rolling(7, min_periods=1).mean()  # Fatiga: media de los últimos 7 días
    serie["Carga Cronica"] = serie["Carga diaria"].rolling(28, min_periods=1).mean()  # Condición: media de 28 días
    serie["ACWR"] = (serie["Carga Aguda"] / serie["Carga Cronica"]).replace([np.inf, -np.inf], np.nan)  # Ratio agudo:crónico

    return serie                                                                   # Devuelve la serie diaria completa


def estado_actual(df, hoy=None):
    """
    Calcula el estado de fatiga vigente a día de hoy sobre el histórico completo.

    Devuelve un diccionario con el ACWR, la etiqueta de riesgo y los días
    transcurridos desde la última actividad registrada.
    """
    hoy = pd.Timestamp(hoy).normalize() if hoy is not None else pd.Timestamp.today().normalize()  # Fecha de referencia

    if df.empty:                                                                   # Sin datos no hay diagnóstico
        return {"acwr": np.nan, "estado": "Sin datos", "dias_inactivo": None}      # Devuelve un estado neutro

    ultima = df["Fecha"].max()                                                     # Fecha de la última actividad
    dias_inactivo = (hoy - ultima.normalize()).days                                # Días transcurridos sin entrenar

    serie = serie_carga_diaria(df, fecha_corte=hoy)                                # Serie diaria hasta la fecha actual
    if serie.empty or serie["Carga Cronica"].iloc[-1] == 0:                        # Sin carga crónica el ratio no aplica
        return {"acwr": np.nan, "estado": "Sin datos suficientes", "dias_inactivo": dias_inactivo}

    acwr = float(serie["ACWR"].iloc[-1])                                           # Último valor del ratio agudo:crónico

    if acwr > 1.5:                                                                 # Umbral superior de sobrecarga
        estado = "ALTO RIESGO DE LESIÓN"                                           # Zona roja de la literatura (Gabbett)
    elif acwr < 0.8:                                                               # Umbral inferior de estímulo
        estado = "CARGA BAJA (desentrenamiento)"                                   # Pérdida progresiva de condición
    else:                                                                          # Franja recomendada 0.8 - 1.5
        estado = "ESTADO ÓPTIMO"                                                   # Zona segura de progresión

    return {"acwr": acwr, "estado": estado, "dias_inactivo": dias_inactivo}        # Devuelve el diagnóstico completo


def calcular_kpis(df, hoy=None):
    """Calcula los indicadores numéricos del panel principal."""
    hoy = pd.Timestamp(hoy).normalize() if hoy is not None else pd.Timestamp.today().normalize()  # Fecha de referencia
    kpis = {"actividades": len(df)}                                                # Número total de actividades

    if df.empty:                                                                   # Protege contra filtros sin resultados
        return kpis                                                                # Devuelve solo el conteo

    ultimos_7 = df[df["Fecha"] >= hoy - pd.Timedelta(days=7)]                      # Ventana móvil de 7 días
    ultimos_28 = df[df["Fecha"] >= hoy - pd.Timedelta(days=28)]                    # Ventana móvil de 28 días

    kpis["km_7d"] = ultimos_7["Distancia_km"].sum()                                # Volumen de la última semana
    kpis["km_28d"] = ultimos_28["Distancia_km"].sum()                              # Volumen del último mesociclo
    kpis["km_total"] = df["Distancia_km"].sum()                                    # Volumen histórico acumulado
    kpis["desnivel_28d"] = ultimos_28["Desnivel positivo"].sum()                   # Desnivel del último mesociclo
    kpis["horas_28d"] = ultimos_28["Minutos"].sum() / 60                           # Horas entrenadas en 28 días
    kpis["cobertura_fc"] = 100 * df["Tiene FC"].mean()                             # Porcentaje real de datos con pulsómetro

    ritmos = df["Ritmo (min/km)"].dropna()                                         # Ritmos válidos tras la limpieza
    kpis["mejor_ritmo"] = ritmos.min() if not ritmos.empty else np.nan             # Mejor ritmo del periodo analizado
    kpis["ritmo_medio"] = ritmos.mean() if not ritmos.empty else np.nan            # Ritmo medio del periodo analizado

    return kpis                                                                    # Devuelve el diccionario de indicadores


def resumen_semanal(df):
    """Agrega el volumen y la carga por semana calendario, por deporte."""
    if df.empty:                                                                   # Sin datos no hay agregación
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    semanal = df.groupby(["Semana", "Tipo de actividad"]).agg(                     # Agrupa por semana y deporte
        Kilometros=("Distancia_km", "sum"),                                        # Volumen semanal en kilómetros
        Carga=("Carga", "sum"),                                                    # Carga total acumulada en la semana
    ).reset_index()                                                                # Devuelve el índice a columnas
    return semanal.round(1)                                                        # Redondea a un decimal para las etiquetas


def resumen_por_tipo(df):
    """Resume actividades, volumen y horas por tipo de deporte."""
    if df.empty:                                                                   # Sin datos no hay resumen
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    resumen = df.groupby("Tipo de actividad").agg(                                 # Agrupa por deporte practicado
        Actividades=("Fecha", "count"),                                            # Número de sesiones
        Kilometros=("Distancia_km", "sum"),                                        # Volumen total acumulado
        Horas=("Minutos", lambda x: x.sum() / 60),                                 # Tiempo total en horas
    ).reset_index().sort_values("Actividades", ascending=False)                    # Ordena de mayor a menor frecuencia

    return resumen.round(1).reset_index(drop=True)                                 # Redondea y reindexa el resultado


def ultimas_actividades(df, n=10):
    """Devuelve las últimas n actividades con las columnas ya formateadas."""
    if df.empty:                                                                   # Sin datos no hay tabla
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    tabla = df.tail(n).sort_values("Fecha", ascending=False).copy()                # Toma las más recientes primero
    tabla["Fecha"] = tabla["Fecha"].dt.strftime("%d/%m/%Y")                        # Formatea la fecha de forma legible
    columnas = ["Fecha", "Tipo de actividad", "Distancia_km", "Minutos",           # Columnas relevantes para el usuario
                "Ritmo (min/km)", "Carga"]
    tabla = tabla[columnas].round(1)                                               # Redondea para evitar decimales largos
    return tabla.rename(columns={"Distancia_km": "Km", "Minutos": "Min"})          # Nombres cortos para la tabla
