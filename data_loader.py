"""
Módulo de Adquisición de Datos del SII "Entrenador IA FCDIA".

Implementa la capa de entrada del sistema con una estrategia híbrida:
    1. Carga inicial (backfill) desde el CSV histórico exportado de Strava.
    2. Sincronización incremental vía API REST de Strava (OAuth 2.0), que
       descarga únicamente las actividades posteriores a la última registrada.

El resultado se persiste en un CSV local que crece con el tiempo, actuando como
capa de almacenamiento del sistema.
"""

import os                                                                          # Acceso a variables de entorno
import time                                                                        # Conversión de fechas a epoch
import pandas as pd                                                                # Manejo de datos tabulares
import requests                                                                    # Cliente HTTP para la API de Strava
from dotenv import load_dotenv                                                     # Carga del archivo .env local

load_dotenv()                                                                      # Lee el archivo .env y lo carga en memoria

RUTA_CSV = "actividades_strava.csv"                                                # CSV histórico exportado de Strava
RUTA_SYNC = "actividades_sincronizadas.csv"                                        # CSV incremental generado por la API

MARGEN_SINCRONIZACION = 3                                                          # Días de solape al descargar novedades

URL_TOKEN = "https://www.strava.com/oauth/token"                                   # Endpoint de renovación de token
URL_ACTIVIDADES = "https://www.strava.com/api/v3/athlete/activities"               # Endpoint de listado de actividades

# Traducción de los tipos de deporte que devuelve la API (inglés) al formato del CSV (español)
TIPOS_API_A_CSV = {
    "Run": "Carrera", "TrailRun": "Carrera", "VirtualRun": "Carrera",              # Variantes de carrera a pie
    "Ride": "Bicicleta", "VirtualRide": "Bicicleta", "MountainBikeRide": "Bicicleta",  # Variantes de ciclismo
    "GravelRide": "Bicicleta", "EBikeRide": "Bicicleta",                           # Más variantes de ciclismo
    "Walk": "Caminata", "Hike": "Senderismo", "Swim": "Natación",                  # Deportes de baja intensidad
    "Workout": "Entrenamiento", "WeightTraining": "Entrenamiento con pesas",       # Entrenamientos sin GPS
    "Rowing": "Remo",                                                              # Remo
}


def credenciales_strava_disponibles():
    """Indica si las tres credenciales de Strava están presentes en el .env."""
    claves = ["STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET", "STRAVA_REFRESH_TOKEN"]  # Variables obligatorias
    return all(os.getenv(k) for k in claves)                                       # True solo si están las tres


def _obtener_access_token():
    """Canjea el refresh token por un access token temporal (válido ~6 horas)."""
    payload = {                                                                    # Cuerpo de la petición OAuth
        "client_id": os.getenv("STRAVA_CLIENT_ID"),                                # Identificador de la aplicación
        "client_secret": os.getenv("STRAVA_CLIENT_SECRET"),                        # Secreto de la aplicación
        "refresh_token": os.getenv("STRAVA_REFRESH_TOKEN"),                        # Token maestro persistente
        "grant_type": "refresh_token",                                             # Tipo de concesión OAuth 2.0
    }
    respuesta = requests.post(URL_TOKEN, data=payload, timeout=30)                 # Petición con verificación SSL activa
    respuesta.raise_for_status()                                                   # Lanza excepción si el token falla
    return respuesta.json().get("access_token")                                    # Devuelve el token de acceso


def _descargar_actividades(access_token, desde_epoch=None, max_paginas=15):
    """
    Descarga actividades paginando la API (200 por página, tope de rate limit).
    Si se indica 'desde_epoch', solo trae actividades posteriores a esa fecha.
    """
    cabecera = {"Authorization": f"Bearer {access_token}"}                         # Autenticación por token Bearer
    acumulado = []                                                                 # Lista de actividades descargadas

    for pagina in range(1, max_paginas + 1):                                       # Recorre las páginas de resultados
        parametros = {"per_page": 200, "page": pagina}                             # Máximo permitido por la API
        if desde_epoch:                                                            # Si hay una fecha de corte
            parametros["after"] = int(desde_epoch)                                 # Solo pide lo posterior a esa fecha

        respuesta = requests.get(URL_ACTIVIDADES, headers=cabecera,                # Petición al listado de actividades
                                 params=parametros, timeout=30)
        respuesta.raise_for_status()                                               # Aborta si la API devuelve error
        lote = respuesta.json()                                                    # Convierte la respuesta a lista JSON

        if not lote:                                                               # Página vacía = no hay más datos
            break                                                                  # Termina la paginación
        acumulado.extend(lote)                                                     # Acumula el lote descargado

    return acumulado                                                               # Devuelve todas las actividades


def _json_a_formato_csv(actividades):
    """Convierte el JSON de la API al mismo esquema de columnas del CSV de Strava."""
    if not actividades:                                                            # Si no llegó ninguna actividad
        return pd.DataFrame()                                                      # Devuelve un DataFrame vacío

    df = pd.json_normalize(actividades)                                            # Aplana el JSON a formato tabular
    tipo_origen = "sport_type" if "sport_type" in df.columns else "type"           # 'type' está deprecado en la API

    salida = pd.DataFrame()                                                        # Contenedor con el esquema del CSV
    salida["Fecha de la actividad"] = df.get("start_date_local")                   # Marca temporal local
    salida["Tipo de actividad"] = df[tipo_origen].map(TIPOS_API_A_CSV).fillna(df[tipo_origen])  # Traduce el deporte
    salida["Distancia.1"] = df.get("distance")                                     # Distancia en metros (igual que el CSV)
    salida["Tiempo en movimiento"] = df.get("moving_time")                         # Tiempo en movimiento en segundos
    salida["Desnivel positivo"] = df.get("total_elevation_gain")                   # Desnivel acumulado en metros
    salida["Ritmo cardiaco promedio"] = df.get("average_heartrate")                # FC media (nula si no hubo pulsómetro)
    salida["Velocidad promedio"] = df.get("average_speed")                         # Velocidad media en m/s

    return salida                                                                  # Devuelve el DataFrame homologado


def sincronizar_con_strava(fecha_ultima_actividad=None):
    """
    Descarga de la API únicamente las actividades posteriores a la última que ya
    se tiene registrada y las guarda de forma acumulativa en RUTA_SYNC.

    Devuelve una tupla (numero_de_actividades_nuevas, mensaje_de_estado).
    """
    if not credenciales_strava_disponibles():                                      # Verifica que existan las claves
        return 0, "Faltan credenciales de Strava en el archivo .env."              # Aborta sin romper la aplicación

    try:
        token = _obtener_access_token()                                            # Renueva el token de acceso
        desde = None                                                               # Por defecto descarga todo
        if fecha_ultima_actividad is not None and pd.notna(fecha_ultima_actividad):  # Si ya hay histórico previo
            # Se retrocede un margen de seguridad porque el parámetro 'after' de la API filtra
            # sobre la hora UTC, mientras que las fechas almacenadas son hora local. Sin este
            # margen, las actividades situadas junto al límite de la ventana se pierden.
            # Los registros repetidos se eliminan después por fecha, así que solapar no duplica.
            corte = pd.Timestamp(fecha_ultima_actividad) - pd.Timedelta(days=MARGEN_SINCRONIZACION)
            desde = corte.timestamp()                                              # Convierte el corte a epoch UTC

        crudas = _descargar_actividades(token, desde_epoch=desde)                  # Descarga el lote incremental
        nuevas = _json_a_formato_csv(crudas)                                       # Homologa al esquema del CSV

        if nuevas.empty:                                                           # Si no hay actividades nuevas
            return 0, "Sin actividades nuevas. Ya estás al día."                   # Informa y termina

        if os.path.exists(RUTA_SYNC):                                              # Si ya existe un archivo de sync
            previas = pd.read_csv(RUTA_SYNC)                                       # Lee lo sincronizado antes
            nuevas = pd.concat([previas, nuevas], ignore_index=True)               # Une lo viejo con lo nuevo
            nuevas = nuevas.drop_duplicates(subset=["Fecha de la actividad"])      # Evita duplicar actividades

        nuevas.to_csv(RUTA_SYNC, index=False)                                      # Persiste el histórico incremental
        return len(crudas), f"Se descargaron {len(crudas)} actividades nuevas."    # Informa el resultado

    except requests.exceptions.HTTPError as error:                                 # Error devuelto por la API
        return 0, f"Strava rechazó la petición ({error.response.status_code}). Revisa tus credenciales."
    except Exception as error:                                                     # Cualquier otro fallo (red, parseo)
        return 0, f"No se pudo sincronizar: {error}"                               # Mensaje amigable sin romper la app


def leer_fuentes_crudas():
    """
    Une el CSV histórico con el CSV de actividades sincronizadas por API.
    Devuelve un único DataFrame crudo, o None si no existe ninguna fuente.
    """
    fuentes = []                                                                   # Lista de DataFrames disponibles

    if os.path.exists(RUTA_CSV):                                                   # Si existe el histórico exportado
        fuentes.append(pd.read_csv(RUTA_CSV, low_memory=False))                    # Lo añade como base

    if os.path.exists(RUTA_SYNC):                                                  # Si existe el archivo sincronizado
        fuentes.append(pd.read_csv(RUTA_SYNC, low_memory=False))                   # Lo añade como complemento

    if not fuentes:                                                                # Si no hay ninguna fuente de datos
        return None                                                                # Señaliza la ausencia de datos

    return pd.concat(fuentes, ignore_index=True)                                   # Devuelve la unión de ambas fuentes
