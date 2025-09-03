import streamlit as st
import pandas as pd
import json, os
from web_ingestor import crawl_web
import db

DEFAULT_PARAMS_PATH = "parametros_logistica.json"

@st.cache_data(show_spinner=False)
def load_params(path=DEFAULT_PARAMS_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "divisor_volumetrico": 5000,
        "clases_por_peso": [
            {"kg_min": 0.0, "clase": "XS"},
            {"kg_min": 0.5, "clase": "S"},
            {"kg_min": 1.0, "clase": "M"},
            {"kg_min": 2.0, "clase": "L"},
            {"kg_min": 5.0, "clase": "XL"},
            {"kg_min": 10.0, "clase": "XXL"},
            {"kg_min": 20.0, "clase": "Oversize"},
        ],
    }


st.set_page_config(page_title="Clasificador Logístico Ripley", page_icon="📦", layout="wide")
st.title("📦 Clasificador de Clase Logística (auto-ingesta web)")
st.caption("Crawling legal (robots.txt), reglas por peso/volumen y diccionario vivo.")

params = load_params()

# Estado inicial para búsqueda web
if "search_query" not in st.session_state:
    st.session_state.search_query = "producto"
if "max_pages" not in st.session_state:
    st.session_state.max_pages = 25
if "delay" not in st.session_state:
    st.session_state.delay = 1.0

# Inicializar base de datos y cargar diccionario
db.init_db()
if "dict_df" not in st.session_state:
    st.session_state.dict_df = db.load_dictionary()

# Sidebar eliminado para interfaz minimalista

st.subheader("📥 Ingesta automática desde la web (crawler)")
st.write("La app buscará en la web páginas que contengan metadatos **JSON-LD Product**.")

st.session_state.search_query = st.text_input(
    "Término de búsqueda web",
    value=st.session_state.search_query,
    help="Se usará DuckDuckGo para descubrir URLs iniciales",
)
colA, colB = st.columns(2)
with colA:
    st.session_state.max_pages = st.number_input(
        "Máx. páginas a explorar",
        value=int(st.session_state.max_pages),
        min_value=5,
        step=5,
    )
with colB:
    st.session_state.delay = st.number_input(
        "Delay entre requests (seg)",
        value=float(st.session_state.delay),
        min_value=0.5,
        step=0.5,
    )

if st.button("🚀 Ejecutar ingesta web ahora", use_container_width=True):
    progress = st.progress(0.0, text=f"Buscando: {st.session_state.search_query}")
    try:
        rows = crawl_web(
            st.session_state.search_query,
            params["clases_por_peso"],
            params["divisor_volumetrico"],
            max_pages=int(st.session_state.max_pages),
            delay=float(st.session_state.delay),
        )
    except Exception as e:
        rows = []
        st.warning(f"Error: {e}")
    progress.progress(1.0, text="Completado.")
    if rows:
        new_df = pd.DataFrame(rows)
        merged = pd.concat([st.session_state.dict_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["hash_row"], keep="first")
        st.session_state.dict_df = merged
        for _, row in new_df.iterrows():
            db.upsert_product(row.to_dict())
        st.success(f"Ingesta completa. Nuevos registros: {len(new_df)} | Total en diccionario: {len(st.session_state.dict_df)}")
    else:
        st.info("No se encontraron productos para la búsqueda indicada.")

st.markdown("Vista del diccionario (primeros 200):")
st.dataframe(st.session_state.dict_df.head(200), use_container_width=True, height=350)

st.subheader("⬇️ Exportar diccionario")
export_df = db.load_dictionary()
csv_data = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", csv_data, file_name="diccionario_logistica.csv", mime="text/csv")
