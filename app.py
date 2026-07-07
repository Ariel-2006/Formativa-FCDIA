"""
Frontend (Streamlit) del Sistema de Información Inteligente "Entrenador IA FCDIA".

Construye la interfaz visual: cabecera, panel de KPIs y pestañas de análisis con
gráficos interactivos (Plotly), además del chat con el agente LLM (Claude). Toda
la lógica de datos vive en 'backend.py'; este archivo solo se ocupa de la
presentación y de la interacción con la IA.
"""

import streamlit as st                                                             # Framework para la interfaz web
import pandas as pd                                                                # Manejo puntual de datos en el frontend
import plotly.express as px                                                        # Gráficos interactivos y modernos
import anthropic                                                                    # Cliente oficial de la API de Claude
import os                                                                           # Acceso a variables de entorno
from dotenv import load_dotenv                                                      # Carga del archivo .env local

import backend                                                                      # Módulo propio con la lógica de datos

# ---------------------------------------------------------
# CONFIGURACIÓN DE ENTORNO SEGURO  (NO MODIFICAR)
# ---------------------------------------------------------
load_dotenv()                                                                       # Lee el archivo .env local y lo carga en memoria
API_KEY_CLAUDE = os.getenv("ANTHROPIC_API_KEY")                                     # Recupera de forma segura la API Key

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA (UX/UI)
# ---------------------------------------------------------
st.set_page_config(page_title="Entrenador IA | FCDIA", page_icon="🏃‍♂️", layout="wide")  # Configura la ventana del navegador

# ---------------------------------------------------------
# ESTILOS PERSONALIZADOS (CSS) PARA UN LOOK MÁS PROFESIONAL
# ---------------------------------------------------------
st.markdown(                                                                        # Inyecta CSS para mejorar la estética
    """
    <style>
        .stMetric { background: #FFFFFF; border: 1px solid #EEE;                    /* Tarjetas de KPI con borde suave */
                    border-radius: 14px; padding: 16px 18px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .titulo-app { background: linear-gradient(90deg,#FF6B00,#FF9E3D);           /* Cabecera con degradado naranja */
                      color: white; padding: 22px 28px; border-radius: 16px;
                      margin-bottom: 8px; }
        .titulo-app h1 { margin: 0; font-size: 30px; }
        .titulo-app p  { margin: 6px 0 0 0; opacity: 0.95; }
    </style>
    """,
    unsafe_allow_html=True,                                                         # Permite renderizar HTML/CSS personalizado
)

# ---------------------------------------------------------
# CABECERA PRINCIPAL
# ---------------------------------------------------------
st.markdown(                                                                        # Renderiza la cabecera con estilo propio
    """
    <div class="titulo-app">
        <h1>🏃‍♂️ Sistema Inteligente: Entrenador IA — FCDIA</h1>
        <p>Monitoreo de carga de entrenamiento, predicción de fatiga y decisiones asistidas por un Agente LLM.</p>
    </div>
    """,
    unsafe_allow_html=True,                                                         # Habilita el bloque HTML de la cabecera
)

# ---------------------------------------------------------
# CARGA DE DATOS (delegada al backend)
# ---------------------------------------------------------
datos = backend.cargar_y_procesar_datos()                                          # Obtiene el DataFrame ya procesado

if datos is None or datos.empty:                                                   # Si no hay archivo o quedó vacío
    st.error("⚠️ Archivo no encontrado o sin datos válidos. "                       # Muestra un aviso amigable
             "Asegúrate de tener 'actividades_strava.csv' en la carpeta del proyecto.")
else:
    estado_riesgo, ratio = backend.predecir_riesgo_fatiga(datos)                   # Ejecuta el componente predictivo
    kpis = backend.calcular_kpis(datos)                                            # Calcula los indicadores clave

    # -----------------------------------------------------
    # PESTAÑAS DE ANÁLISIS
    # -----------------------------------------------------
    tab_resumen, tab_carga, tab_ritmo, tab_tipo = st.tabs(                         # Crea las 4 pestañas de análisis
        ["📊 Resumen", "🔥 Carga y Fatiga", "📏 Distancia y Ritmo", "🏃 Por Tipo"]
    )

    # ============ PESTAÑA 1: RESUMEN ============
    with tab_resumen:                                                              # Contenido de la pestaña Resumen
        st.subheader("📊 Panel de Rendimiento")                                    # Título de la sección

        c1, c2, c3, c4 = st.columns(4)                                             # Divide el panel en 4 columnas
        with c1:                                                                   # Primera métrica
            st.metric("Distancia Total", f"{kpis.get('distancia_total_km', 0):.1f} km")   # Distancia histórica
        with c2:                                                                   # Segunda métrica
            st.metric("Actividades", f"{kpis.get('num_actividades', 0)}")          # Número de sesiones
        with c3:                                                                   # Tercera métrica
            st.metric("FC Promedio", f"{kpis.get('fc_promedio', 0):.0f} ppm")      # Frecuencia cardiaca media
        with c4:                                                                   # Cuarta métrica
            st.metric("Ratio de Fatiga (A:C)", f"{ratio:.2f}")                     # Componente predictivo principal

        st.markdown("### 🩺 Diagnóstico del sistema")                              # Subtítulo del diagnóstico
        if ratio > 1.5:                                                            # Zona de riesgo alto
            st.error(f"Predicción: {estado_riesgo} — reduce la intensidad y prioriza recuperación.")  # Alerta roja
        elif ratio < 0.8:                                                          # Zona de baja carga
            st.warning(f"Predicción: {estado_riesgo} — puedes aumentar el volumen de forma gradual.")  # Alerta amarilla
        else:                                                                      # Zona segura
            st.success(f"Predicción: {estado_riesgo} — mantén la progresión actual.")   # Alerta verde

        st.markdown("### 📈 Evolución de la Carga de Entrenamiento")               # Subtítulo del gráfico principal
        if "Carga Estimada" in datos.columns:                                      # Solo grafica si existe la carga
            fig_carga = px.bar(                                                     # Gráfico de barras de la carga
                datos.tail(30), x="Fecha de la actividad", y="Carga Estimada",
                color="Carga Estimada", color_continuous_scale="Oranges",
                title="Carga de Entrenamiento (últimas 30 actividades)",
            )
            st.plotly_chart(fig_carga, width="stretch")                            # Renderiza el gráfico

    # ============ PESTAÑA 2: CARGA Y FATIGA ============
    with tab_carga:                                                               # Contenido de la pestaña Carga y Fatiga
        st.subheader("🔥 Carga Aguda vs Crónica (ACWR)")                          # Título de la sección
        serie = backend.serie_carga_acwr(datos)                                   # Calcula las medias móviles y el ACWR

        if "ACWR" in serie.columns:                                               # Solo si se pudo calcular el ACWR
            fig_ac = px.line(                                                      # Líneas de carga aguda vs crónica
                serie, x="Fecha", y=["Carga Aguda", "Carga Cronica"],
                title="Fatiga (7) vs Fitness (28)",
                labels={"value": "Carga Estimada", "variable": "Serie"},
            )
            st.plotly_chart(fig_ac, width="stretch")                              # Renderiza el gráfico comparativo

            fig_ratio = px.line(                                                   # Línea de evolución del ratio ACWR
                serie, x="Fecha", y="ACWR", title="Evolución del Ratio de Fatiga (A:C)",
            )
            fig_ratio.add_hline(y=1.5, line_dash="dash", line_color="red",         # Umbral de riesgo alto
                                annotation_text="Riesgo alto (1.5)")
            fig_ratio.add_hline(y=0.8, line_dash="dash", line_color="orange",      # Umbral de baja carga
                                annotation_text="Sub-entreno (0.8)")
            st.plotly_chart(fig_ratio, width="stretch")                           # Renderiza el gráfico del ratio
        else:                                                                      # Si no hay datos de carga
            st.info("No hay suficiente 'Carga Estimada' para calcular el ACWR.")   # Mensaje informativo

        st.markdown("### 🗓️ Carga acumulada por semana")                         # Subtítulo de la carga semanal
        semanal = backend.carga_por_semana(datos)                                 # Agrega la carga por semana
        if not semanal.empty and "Carga Estimada" in semanal.columns:             # Solo si hay datos semanales
            fig_sem = px.bar(                                                      # Barras de carga semanal
                semanal, x="Semana", y="Carga Estimada",
                color="Carga Estimada", color_continuous_scale="Reds",
                title="Carga total por semana",
            )
            st.plotly_chart(fig_sem, width="stretch")                             # Renderiza el gráfico semanal
        else:                                                                      # Si no hay agregado semanal
            st.info("No hay datos semanales disponibles.")                         # Mensaje informativo

    # ============ PESTAÑA 3: DISTANCIA Y RITMO ============
    with tab_ritmo:                                                              # Contenido de la pestaña Distancia y Ritmo
        st.subheader("📏 Distancia, Ritmo y Frecuencia Cardiaca")                 # Título de la sección
        col_a, col_b = st.columns(2)                                              # Dos columnas de gráficos

        with col_a:                                                               # Columna izquierda
            if {"Fecha", "Distancia_km"}.issubset(datos.columns):                 # Requiere fecha y distancia
                fig_dist = px.area(                                                # Área de distancia en el tiempo
                    datos, x="Fecha", y="Distancia_km",
                    title="Distancia por actividad (km)",
                )
                st.plotly_chart(fig_dist, width="stretch")                        # Renderiza el gráfico de distancia
            if "Distancia_km" in datos.columns:                                   # Requiere la distancia
                fig_hist = px.histogram(                                           # Histograma de distancias
                    datos, x="Distancia_km", nbins=15,
                    title="Distribución de distancias",
                )
                st.plotly_chart(fig_hist, width="stretch")                        # Renderiza el histograma

        with col_b:                                                               # Columna derecha
            if {"Fecha", "Ritmo cardiaco promedio"}.issubset(datos.columns):      # Requiere fecha y FC
                fig_fc = px.line(                                                  # Línea de FC en el tiempo
                    datos, x="Fecha", y="Ritmo cardiaco promedio",
                    title="Frecuencia cardiaca por actividad (ppm)",
                )
                st.plotly_chart(fig_fc, width="stretch")                          # Renderiza el gráfico de FC
            if {"Fecha", "Ritmo (min/km)"}.issubset(datos.columns):               # Requiere fecha y ritmo
                fig_pace = px.line(                                                # Línea de ritmo en el tiempo
                    datos.dropna(subset=["Ritmo (min/km)"]),
                    x="Fecha", y="Ritmo (min/km)",
                    title="Ritmo por actividad (min/km)",
                )
                st.plotly_chart(fig_pace, width="stretch")                        # Renderiza el gráfico de ritmo

    # ============ PESTAÑA 4: POR TIPO ============
    with tab_tipo:                                                               # Contenido de la pestaña Por Tipo
        st.subheader("🏃 Análisis por Tipo de Actividad")                        # Título de la sección
        resumen_tipo = backend.resumen_por_tipo(datos)                            # Agrega los datos por tipo

        if resumen_tipo.empty:                                                    # Si no existe la columna de tipo
            st.info("La columna 'Tipo de actividad' no está disponible en el CSV.")  # Mensaje informativo
        else:                                                                     # Si hay datos por tipo
            col_p, col_q = st.columns(2)                                          # Dos columnas de gráficos
            with col_p:                                                           # Columna izquierda
                fig_pie = px.pie(                                                  # Distribución de actividades por tipo
                    resumen_tipo, names="Tipo de actividad", values="Actividades",
                    title="Proporción de actividades",
                )
                st.plotly_chart(fig_pie, width="stretch")                        # Renderiza el gráfico circular
            with col_q:                                                           # Columna derecha
                if "Distancia_km" in resumen_tipo.columns:                        # Solo si hay distancia agregada
                    fig_bar = px.bar(                                              # Distancia total por tipo
                        resumen_tipo, x="Tipo de actividad", y="Distancia_km",
                        color="Tipo de actividad", title="Distancia total por tipo (km)",
                    )
                    st.plotly_chart(fig_bar, width="stretch")                    # Renderiza el gráfico de barras
            st.dataframe(resumen_tipo, width="stretch")                          # Muestra la tabla resumen por tipo

    # ---------------------------------------------------------
    # AGENTE LLM (CLAUDE) Y CHATBOT CON .ENV  (SIN CAMBIOS)
    # ---------------------------------------------------------
    st.markdown("---")                                                            # Separador visual
    st.subheader("🤖 Entrenador IA (Consulta y Recomendaciones)")                 # Título del chat

    # Inicializa el historial del chat en la memoria de la sesión
    if "mensajes" not in st.session_state:                                        # Comprueba si ya existe el historial
        st.session_state.mensajes = []                                            # Crea lista vacía inicial

    # Muestra los mensajes previos en pantalla
    for msg in st.session_state.mensajes:                                         # Recorre el historial guardado
        with st.chat_message(msg["role"]):                                        # Abre el globo del mensaje
            st.markdown(msg["content"])                                           # Imprime el texto

    # Captura lo que escribe el usuario
    if prompt := st.chat_input("Ej: ¿Qué me recomiendas entrenar mañana?"):       # Espera la entrada del usuario

        st.session_state.mensajes.append({"role": "user", "content": prompt})     # Guarda el mensaje del usuario
        with st.chat_message("user"):                                             # Abre el globo del usuario
            st.markdown(prompt)                                                   # Renderiza el texto del usuario

        # Comprobamos si la API Key se cargó correctamente desde el entorno
        if not API_KEY_CLAUDE:                                                    # Verifica que exista la clave
            st.error("❌ Error: No se encontró la variable 'ANTHROPIC_API_KEY' en el archivo .env local.")  # Aviso de error
        else:
            try:
                # Usamos la variable segura cargada al inicio
                cliente = anthropic.Anthropic(api_key=API_KEY_CLAUDE)             # Inicializa el cliente con la clave del .env

                contexto_oculto = f"""
                Eres un entrenador deportivo profesional analizando a un atleta amateur.
                DATO CRÍTICO ACTUAL: El sistema predictivo marca un ratio de fatiga de {ratio:.2f}.
                Estado del atleta: {estado_riesgo}. 
                Responde directamente al usuario considerando este riesgo. Sé breve, amigable y da una instrucción clara.
                Sé conciso. No hagas introducciones largas y ve directo a la recomendación.
                """

                with st.chat_message("assistant"):                               # Abre el globo del asistente
                    respuesta = cliente.messages.create(                          # Llama a la API de Claude
                        model="claude-haiku-4-5-20251001",                       # Modelo rápido y económico
                        max_tokens=600,
                        system=contexto_oculto,                                  # Inyecta el sistema inteligente
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
                    )
                    texto_respuesta = respuesta.content[0].text                  # Extrae el texto de la respuesta
                    st.markdown(texto_respuesta)                                 # Muestra respuesta de la IA

                st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})  # Guarda en memoria
            except Exception as e:                                               # Captura cualquier error de conexión
                st.error(f"Error de conexión con la IA: {e}")                    # Muestra el error de forma amigable