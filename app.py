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
    st.session_state.search_query = ""

# Inicializar base de datos y cargar diccionario
db.init_db()
if "dict_df" not in st.session_state:
    st.session_state.dict_df = db.load_dictionary()

MAX_PAGES = 25
DELAY = 1.0

def run_ingesta():
    query = st.session_state.search_query.strip()
    if not query:
        return
    with st.spinner(f"Buscando: {query}"):
        try:
            rows = crawl_web(
                query,
                params["clases_por_peso"],
                params["divisor_volumetrico"],
                max_pages=MAX_PAGES,
                delay=DELAY,
            )
        except Exception as e:
            rows = []
            st.warning(f"Error: {e}")
    if rows:
        new_df = pd.DataFrame(rows)
        merged = pd.concat([st.session_state.dict_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["hash_row"], keep="first")
        st.session_state.dict_df = merged
        for _, row in new_df.iterrows():
            db.upsert_product(row.to_dict())
        st.success(
            f"Ingesta completa. Nuevos registros: {len(new_df)} | Total en diccionario: {len(st.session_state.dict_df)}"
        )
    else:
        st.info("No se encontraron productos para la búsqueda indicada.")

st.text_input(
    "Nombre del producto",
    key="search_query",
    on_change=run_ingesta,
    placeholder="Ej: PS5",
)

st.markdown("Vista del diccionario (primeros 200):")
st.dataframe(st.session_state.dict_df.head(200), use_container_width=True, height=350)

st.subheader("⬇️ Exportar diccionario")
export_df = db.load_dictionary()
csv_data = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", csv_data, file_name="diccionario_logistica.csv", mime="text/csv")
