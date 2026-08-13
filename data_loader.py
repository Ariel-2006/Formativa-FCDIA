"""
Módulo de Adquisición de Datos del SII "Entrenador IA FCDIA".
Implementa la capa de entrada del sistema con una estrategia híbrida:
1. Carga inicial (backfill) desde el CSV histórico exportado de Strava.
2. Sincronización incremental vía API REST de Strava (OAuth 2.0), que
   descarga únicamente las actividades posteriores a la última registrada.
El resultado se persiste en Supabase (antes en un CSV local), actuando como
capa de almacenamiento que sobrevive a los reinicios del entorno en la nube.
"""

import os                                     # Acceso a variables de entorno
import pandas as pd                           # Manejo de datos tabulares
import requests                               # Cliente HTTP para la API de Strava
from dotenv import load_dotenv                # Carga del archivo .env local
from supabase import create_client, Client    # Cliente de la base de datos en la nube

load_dotenv()  # Lee el .env local; en la nube no hace nada (las llaves ya están en Secrets)

RUTA_CSV = "actividades_strava.csv"  # CSV histórico exportado de Strava (se queda en el repo)
MARGEN_SINCRONIZACION = 3            # Días de solape al descargar novedades
URL_TOKEN = "https://www.strava.com/oauth/token"
URL_ACTIVIDADES = "https://www.strava.com/api/v3/athlete/activities"

TABLA_SYNC = "actividades_sincronizadas"  # Tabla de Supabase que reemplaza al CSV incremental

# Traduce entre el nombre de columna que usa el DataFrame interno y el de la tabla en Supabase
MAPA_COLUMNAS_SYNC = {
    "Fecha de la actividad": "fecha_actividad",
    "Tipo de actividad": "tipo_actividad",
    "Distancia.1": "distancia_m",
    "Tiempo en movimiento": "tiempo_movimiento_s",
    "Desnivel positivo": "desnivel_positivo_m",
    "Ritmo cardiaco promedio": "fc_promedio",
    "Velocidad promedio": "velocidad_promedio",
}

# Traducción de los tipos de deporte que devuelve la API (inglés) al formato del CSV (español)
TIPOS_API_A_CSV = {
    "Run": "Carrera", "TrailRun": "Carrera", "VirtualRun": "Carrera",
    "Ride": "Bicicleta", "VirtualRide": "Bicicleta", "MountainBikeRide": "Bicicleta",
    "GravelRide": "Bicicleta", "EBikeRide": "Bicicleta",
    "Walk": "Caminata", "Hike": "Senderismo", "Swim": "Natación",
    "Workout": "Entrenamiento", "WeightTraining": "Entrenamiento con pesas",
    "Rowing": "Remo",
}

_supabase: Client | None = None  # Cliente cacheado; se crea una sola vez por sesión

def _cliente_supabase():
    """Crea (o reutiliza) el cliente de Supabase a partir de las credenciales del entorno."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase

def credenciales_strava_disponibles():
    """Indica si las tres credenciales de Strava están presentes en el .env."""
    claves = ["STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN"]
    return all(os.getenv(k) for k in claves)

def _obtener_access_token():
    """Canjea el refresh token por un access token temporal (válido ~6 horas)."""
    payload = {
        "client_id": os.getenv("STRAVA_CLIENT_ID"),
        "client_secret": os.getenv("STRAVA_CLIENT_SECRET"),
        "refresh_token": os.getenv("STRAVA_REFRESH_TOKEN"),
        "grant_type": "refresh_token",
    }
    respuesta = requests.post(URL_TOKEN, data=payload, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json().get("access_token")

def _descargar_actividades(access_token, desde_epoch=None, max_paginas=15):
    """
    Descarga actividades paginando la API (200 por página, tope de rate limit).
    Si se indica 'desde_epoch', solo trae actividades posteriores a esa fecha.
    """
    cabecera = {"Authorization": f"Bearer {access_token}"}
    acumulado = []
    for pagina in range(1, max_paginas + 1):
        parametros = {"per_page": 200, "page": pagina}
        if desde_epoch:
            parametros["after"] = int(desde_epoch)
        respuesta = requests.get(URL_ACTIVIDADES, headers=cabecera, params=parametros, timeout=30)
        respuesta.raise_for_status()
        lote = respuesta.json()
        if not lote:
            break
        acumulado.extend(lote)
    return acumulado

def _json_a_formato_csv(actividades):
    """Convierte el JSON de la API al mismo esquema de columnas que usaba el CSV."""
    if not actividades:
        return pd.DataFrame()
    df = pd.json_normalize(actividades)
    tipo_origen = "sport_type" if "sport_type" in df.columns else "type"
    salida = pd.DataFrame()
    salida["Fecha de la actividad"] = df.get("start_date_local")
    salida["Tipo de actividad"] = df[tipo_origen].map(TIPOS_API_A_CSV).fillna(df[tipo_origen])
    salida["Distancia.1"] = df.get("distance")
    salida["Tiempo en movimiento"] = df.get("moving_time")
    salida["Desnivel positivo"] = df.get("total_elevation_gain")
    salida["Ritmo cardiaco promedio"] = df.get("average_heartrate")
    salida["Velocidad promedio"] = df.get("average_speed")
    return salida

def _df_a_registros_sync(df):
    """Traduce el DataFrame homologado a una lista de diccionarios lista para Supabase."""
    tabla = df.rename(columns=MAPA_COLUMNAS_SYNC)[list(MAPA_COLUMNAS_SYNC.values())]
    registros = tabla.to_dict(orient="records")
    # to_dict() conserva los NaN como float('nan'); hay que limpiarlos aquí, ya como
    # objetos Python sueltos, porque dentro de una columna float64 el .where(..., None)
    # no funciona: pandas no puede guardar None en una columna numérica y lo revierte a NaN.
    return [
        {clave: (None if isinstance(valor, float) and pd.isna(valor) else valor)
         for clave, valor in registro.items()}
        for registro in registros
    ]

def _leer_sync_supabase():
    """Descarga lo sincronizado hasta ahora y lo devuelve con los nombres de columna originales."""
    respuesta = _cliente_supabase().table(TABLA_SYNC).select("*").execute()
    if not respuesta.data:
        return pd.DataFrame()
    tabla = pd.DataFrame(respuesta.data)
    mapa_inverso = {v: k for k, v in MAPA_COLUMNAS_SYNC.items()}
    tabla = tabla.rename(columns=mapa_inverso)
    return tabla[[c for c in MAPA_COLUMNAS_SYNC.keys() if c in tabla.columns]]

def sincronizar_con_strava(fecha_ultima_actividad=None):
    """
    Descarga de la API únicamente las actividades posteriores a la última que ya
    se tiene registrada y las guarda en Supabase mediante upsert. Como
    'fecha_actividad' es columna única, un registro repetido se sobrescribe en
    vez de duplicarse — el mismo efecto que antes lograba drop_duplicates().
    Devuelve una tupla (numero_de_actividades_nuevas, mensaje_de_estado).
    """
    if not credenciales_strava_disponibles():
        return 0, "Faltan credenciales de Strava en el archivo .env."
    try:
        token = _obtener_access_token()
        desde = None
        if fecha_ultima_actividad is not None and pd.notna(fecha_ultima_actividad):
            # Se retrocede un margen de seguridad porque el parámetro 'after' de la API filtra
            # sobre la hora UTC, mientras que las fechas almacenadas son hora local. El upsert
            # por 'fecha_actividad' evita que este solape genere duplicados.
            corte = pd.Timestamp(fecha_ultima_actividad) - pd.Timedelta(days=MARGEN_SINCRONIZACION)
            desde = corte.timestamp()
        crudas = _descargar_actividades(token, desde_epoch=desde)
        nuevas = _json_a_formato_csv(crudas)
        if nuevas.empty:
            return 0, "Sin actividades nuevas. Ya estás al día."
        registros = _df_a_registros_sync(nuevas)
        _cliente_supabase().table(TABLA_SYNC).upsert(registros, on_conflict="fecha_actividad").execute()
        return len(crudas), f"Se descargaron {len(crudas)} actividades nuevas."
    except requests.exceptions.HTTPError as error:
        return 0, f"Strava rechazó la petición ({error.response.status_code}). Revisa tus credenciales."
    except Exception as error:
        return 0, f"No se pudo sincronizar: {error}"

def leer_fuentes_crudas():
    """
    Une el CSV histórico (local, en el repo) con las actividades sincronizadas
    (Supabase). Devuelve un único DataFrame crudo, o None si no hay ninguna fuente.
    """
    # El export histórico trae 103 columnas y la sincronización solo 7. Recortando
    # ambas fuentes al mismo esquema se evita generar columnas vacías al unirlas.
    columnas_utiles = [
        "Fecha de la actividad", "Tipo de actividad", "Distancia", "Distancia.1",
        "Tiempo en movimiento", "Desnivel positivo", "Ritmo cardiaco promedio",
        "Velocidad promedio",
    ]
    fuentes = []

    if os.path.exists(RUTA_CSV):
        datos = pd.read_csv(RUTA_CSV, low_memory=False)
        presentes = [c for c in columnas_utiles if c in datos.columns]
        datos = datos[presentes].dropna(axis=1, how="all")
        if not datos.empty:
            fuentes.append(datos)

    datos_sync = _leer_sync_supabase()
    if not datos_sync.empty:
        presentes = [c for c in columnas_utiles if c in datos_sync.columns]
        datos_sync = datos_sync[presentes].dropna(axis=1, how="all")
        if not datos_sync.empty:
            fuentes.append(datos_sync)

    if not fuentes:
        return None
    unido = pd.concat(fuentes, ignore_index=True)
    for columna in columnas_utiles:
        if columna not in unido.columns:
            unido[columna] = pd.NA
    return unido