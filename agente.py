"""
Módulo del Agente LLM del SII "Entrenador IA FCDIA".

Gestiona la capa conversacional del sistema:
    - Persiste el historial de chat en disco, de modo que el contexto sobreviva
      al cierre de la aplicación (st.session_state se borra al recargar).
    - Mantiene un diario de estado físico que se inyecta como hecho estructurado.
    - Construye el contexto del sistema con los indicadores y la predicción
      calculados por los demás módulos, de forma que el agente interprete datos
      reales y no opere como un chatbot aislado.
"""

import os                                                                          # Acceso a variables de entorno y ficheros
import json                                                                        # Serialización del historial y del diario
from datetime import date                                                          # Registro de la fecha del diario

RUTA_HISTORIAL = "historial_chat.json"                                             # Fichero de memoria conversacional
RUTA_DIARIO = "diario_estado.json"                                                 # Fichero del diario de estado físico

MAX_MENSAJES = 20                                                                  # Ventana de historial enviada al modelo
MAX_ENTRADAS_DIARIO = 7                                                            # Últimos días de estado que se inyectan

MODELO = "claude-haiku-4-5-20251001"                                               # Modelo rápido y económico de Anthropic


# ---------------------------------------------------------
# PERSISTENCIA DE LA MEMORIA CONVERSACIONAL
# ---------------------------------------------------------
def cargar_historial():
    """Recupera del disco el historial de chat de sesiones anteriores."""
    if not os.path.exists(RUTA_HISTORIAL):                                         # Si aún no existe el fichero
        return []                                                                  # Devuelve un historial vacío
    try:
        with open(RUTA_HISTORIAL, "r", encoding="utf-8") as fichero:               # Abre el fichero de memoria
            return json.load(fichero)                                              # Devuelve la lista de mensajes
    except (json.JSONDecodeError, OSError):                                        # Fichero corrupto o ilegible
        return []                                                                  # Reinicia el historial sin romper la app


def guardar_historial(mensajes):
    """Persiste el historial de chat en disco tras cada intercambio."""
    try:
        with open(RUTA_HISTORIAL, "w", encoding="utf-8") as fichero:               # Abre el fichero en modo escritura
            json.dump(mensajes, fichero, ensure_ascii=False, indent=2)             # Vuelca el historial en formato JSON
    except OSError:                                                                # Fallo de escritura en disco
        pass                                                                       # La conversación continúa igualmente


def borrar_historial():
    """Elimina la memoria conversacional almacenada."""
    if os.path.exists(RUTA_HISTORIAL):                                             # Solo si el fichero existe
        os.remove(RUTA_HISTORIAL)                                                  # Borra el fichero del disco


# ---------------------------------------------------------
# DIARIO DE ESTADO FÍSICO
# ---------------------------------------------------------
def cargar_diario():
    """Recupera el diario de estado físico registrado por el deportista."""
    if not os.path.exists(RUTA_DIARIO):                                            # Si aún no existe el fichero
        return []                                                                  # Devuelve un diario vacío
    try:
        with open(RUTA_DIARIO, "r", encoding="utf-8") as fichero:                  # Abre el fichero del diario
            return json.load(fichero)                                              # Devuelve la lista de registros
    except (json.JSONDecodeError, OSError):                                        # Fichero corrupto o ilegible
        return []                                                                  # Reinicia el diario sin romper la app


def registrar_estado(estado, nota=""):
    """
    Añade o actualiza el estado físico del día en el diario. Este dato se inyecta
    después como hecho estructurado, en lugar de confiar en que el modelo lo
    deduzca del historial conversacional.
    """
    diario = cargar_diario()                                                       # Recupera el diario existente
    hoy = date.today().isoformat()                                                 # Fecha actual en formato ISO
    diario = [d for d in diario if d.get("fecha") != hoy]                          # Elimina el registro previo del día
    diario.append({"fecha": hoy, "estado": estado, "nota": nota})                  # Añade el estado actualizado
    diario = sorted(diario, key=lambda d: d["fecha"])[-30:]                        # Conserva el último mes de registros

    try:
        with open(RUTA_DIARIO, "w", encoding="utf-8") as fichero:                  # Abre el fichero en modo escritura
            json.dump(diario, fichero, ensure_ascii=False, indent=2)               # Persiste el diario actualizado
    except OSError:                                                                # Fallo de escritura en disco
        pass                                                                       # El registro se pierde sin romper la app

    return diario                                                                  # Devuelve el diario ya actualizado


# ---------------------------------------------------------
# CONSTRUCCIÓN DEL CONTEXTO DEL SISTEMA
# ---------------------------------------------------------
def construir_contexto(kpis, diagnostico, prediccion, entrenamiento, diario):
    """
    Ensambla el prompt de sistema con la salida real de los módulos analíticos.

    El agente recibe los indicadores, el resultado del componente predictivo y su
    fiabilidad declarada, de modo que sus recomendaciones se apoyen en los datos
    del sistema y no en suposiciones.
    """
    lineas = [                                                                     # Bloque de rol e instrucciones
        "Eres un entrenador deportivo profesional que asesora a un atleta amateur.",
        "Recibes los indicadores calculados por un sistema de análisis de datos de Strava.",
        "Responde en español, de forma breve, concreta y sin introducciones largas.",
        "",
        "=== ESTADO ACTUAL DEL ATLETA (datos del sistema) ===",
    ]

    acwr = diagnostico.get("acwr")                                                 # Ratio agudo:crónico vigente
    if acwr is not None and acwr == acwr:                                          # Comprueba que no sea un valor NaN
        lineas.append(f"- Ratio de carga aguda:crónica (ACWR): {acwr:.2f} → {diagnostico['estado']}.")
        lineas.append("  Interpretación: por debajo de 0.8 hay desentrenamiento; por encima de 1.5, riesgo de lesión.")
    else:                                                                          # Si el indicador no pudo calcularse
        lineas.append("- ACWR: no disponible por falta de datos recientes.")

    dias = diagnostico.get("dias_inactivo")                                        # Días transcurridos sin entrenar
    if dias is not None:                                                           # Solo si se pudo calcular
        lineas.append(f"- Días desde la última actividad registrada: {dias}.")

    lineas += [                                                                    # Indicadores de volumen y calidad de datos
        f"- Volumen últimos 7 días: {kpis.get('km_7d', 0):.1f} km.",
        f"- Volumen últimos 28 días: {kpis.get('km_28d', 0):.1f} km.",
        f"- Horas entrenadas en 28 días: {kpis.get('horas_28d', 0):.1f} h.",
        f"- Cobertura de pulsómetro en el histórico: {kpis.get('cobertura_fc', 0):.0f} %.",
    ]

    lineas.append("")                                                              # Separador del bloque predictivo
    lineas.append("=== COMPONENTE PREDICTIVO (regresión por mesociclos) ===")

    if prediccion and entrenamiento:                                               # Solo si el modelo pudo entrenarse
        lineas.append(f"- Ritmo de competición estimado: {prediccion['ritmo']:.2f} min/km.")
        lineas.append(f"- Tiempo proyectado en {prediccion['distancia']:.1f} km: {prediccion['tiempo_texto']}.")
        lineas.append(f"- Fiabilidad del modelo: R²={entrenamiento['r2']:.2f}, "
                      f"error medio {entrenamiento['mae']:.2f} min/km "
                      f"(entrenado con {entrenamiento['n_mesociclos']} mesociclos).")
        lineas.append("- Limitación: el objetivo es el mejor esfuerzo del bloque, no un tiempo oficial de competencia.")
    else:                                                                          # Sin muestra suficiente para predecir
        lineas.append("- Sin predicción disponible: no hay carreras recientes suficientes en el bloque actual.")

    if diario:                                                                     # Bloque del diario de estado físico
        lineas.append("")                                                          # Separador visual
        lineas.append("=== DIARIO DE ESTADO FÍSICO (declarado por el atleta) ===")
        for registro in diario[-MAX_ENTRADAS_DIARIO:]:                             # Recorre los últimos días registrados
            nota = f" — {registro['nota']}" if registro.get("nota") else ""        # Añade la nota libre si existe
            lineas.append(f"- {registro['fecha']}: {registro['estado']}{nota}")    # Formatea cada entrada del diario
        lineas.append("IMPORTANTE: prioriza estos estados declarados sobre los indicadores. "
                      "Si el atleta ha estado enfermo o lesionado en los últimos días, tenlo en cuenta "
                      "aunque él proponga una sesión exigente.")

    lineas += [                                                                    # Reglas finales de comportamiento
        "",
        "=== REGLAS ===",
        "1. Fundamenta cada recomendación en los indicadores anteriores, citando el dato concreto.",
        "2. Nunca inventes datos que no aparezcan en este contexto.",
        "3. Recuerda que eres un apoyo a la decisión: la decisión final es del atleta.",
        "4. Si detectas riesgo de lesión o el atleta declara estar enfermo, dilo con claridad.",
    ]

    return "\n".join(lineas)                                                       # Devuelve el prompt de sistema completo


def consultar_agente(cliente, contexto, mensajes):
    """
    Envía la conversación al modelo junto con el contexto del sistema.
    Trunca el historial para acotar el consumo de tokens en sesiones largas.
    """
    recientes = mensajes[-MAX_MENSAJES:]                                           # Ventana de conversación reciente
    respuesta = cliente.messages.create(                                           # Llamada a la API de Anthropic
        model=MODELO,                                                              # Modelo configurado para el agente
        max_tokens=700,                                                            # Límite de longitud de la respuesta
        system=contexto,                                                           # Inyecta los datos del sistema
        messages=[{"role": m["role"], "content": m["content"]} for m in recientes],  # Historial conversacional
    )
    return respuesta.content[0].text                                               # Devuelve el texto de la respuesta
