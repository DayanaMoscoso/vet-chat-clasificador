import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Clasificador de Mensajes - Clínica Veterinaria",
    page_icon="🐾",
    layout="centered"
)

# ============================================
# CARGAR MODELO Y VECTORIZADOR (una sola vez)
# ============================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_clasificador.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return modelo, tfidf

modelo, tfidf = cargar_modelo()

# ============================================
# MISMA FUNCIÓN DE LIMPIEZA USADA EN COLAB
# (debe ser IDÉNTICA a la que usaste para entrenar)
# ============================================
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ============================================
# HISTORIAL EN MEMORIA (dura mientras la pestaña esté abierta)
# ============================================
if "historial" not in st.session_state:
    st.session_state.historial = []

# ============================================
# ENCABEZADO
# ============================================
st.title("🐾 Clasificador de Mensajes")
st.caption("Clínica Veterinaria — Proyecto Capstone")
st.write("Escribe o pega un mensaje del chat para ver a qué categoría pertenece.")

# ============================================
# ENTRADA DE MENSAJE
# ============================================
mensaje = st.text_area("Mensaje del cliente:", height=100, placeholder="Ej: Mi perro no ha comido en dos días y está muy decaído")

col1, col2 = st.columns([1, 1])
with col1:
    clasificar = st.button("🔍 Clasificar mensaje", use_container_width=True, type="primary")
with col2:
    limpiar = st.button("🗑️ Limpiar historial", use_container_width=True)

if limpiar:
    st.session_state.historial = []
    st.rerun()

# ============================================
# CLASIFICACIÓN
# ============================================
if clasificar:
    if mensaje.strip() == "":
        st.warning("Por favor escribe un mensaje antes de clasificar.")
    else:
        texto_limpio = limpiar_texto(mensaje)
        vector = tfidf.transform([texto_limpio])

        prediccion = modelo.predict(vector)[0]

        # Probabilidad / confianza (si el modelo la soporta)
        try:
            probas = modelo.predict_proba(vector)[0]
            clases = modelo.classes_
            confianza = max(probas) * 100
            tabla_probas = pd.DataFrame({
                "Categoría": clases,
                "Probabilidad (%)": (probas * 100).round(2)
            }).sort_values("Probabilidad (%)", ascending=False)
        except AttributeError:
            confianza = None
            tabla_probas = None

        # Mostrar resultado principal
        st.divider()
        st.subheader("Resultado de la clasificación")

        if confianza is not None:
            st.success(f"**Categoría detectada:** {prediccion}  \n**Confianza:** {confianza:.1f}%")
        else:
            st.success(f"**Categoría detectada:** {prediccion}")

        if tabla_probas is not None:
            with st.expander("Ver probabilidades por categoría"):
                st.dataframe(tabla_probas, hide_index=True, use_container_width=True)

        # Guardar en historial
        st.session_state.historial.insert(0, {
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Mensaje": mensaje,
            "Categoría": prediccion,
            "Confianza (%)": f"{confianza:.1f}" if confianza is not None else "N/A"
        })

# ============================================
# HISTORIAL DE LA SESIÓN
# ============================================
if st.session_state.historial:
    st.divider()
    st.subheader("Historial de esta sesión")
    df_historial = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_historial, hide_index=True, use_container_width=True)
