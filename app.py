\
import streamlit as st
import pandas as pd
import json, os, time, io, yaml
import threading
from web_ingestor import crawl_domain
from scheduler import schedule_crawl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from rapidfuzz import process, fuzz
import db

DEFAULT_PARAMS_PATH = "parametros_logistica.json"
DEFAULT_SOURCES_YML = "sources.yml"

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

@st.cache_data(show_spinner=False)
def load_sources(path=DEFAULT_SOURCES_YML):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
            return y.get("sources", []), y.get("max_pages_per_domain", 25), y.get("delay_seconds", 1.0)
    return [], 25, 1.0

def save_params(params, path=DEFAULT_PARAMS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

def save_sources(sources, max_pages, delay, path=DEFAULT_SOURCES_YML):
    y = {"sources": sources, "max_pages_per_domain": max_pages, "delay_seconds": delay}
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))

st.set_page_config(page_title="Clasificador Logístico Ripley", page_icon="📦", layout="wide")
st.title("📦 Clasificador de Clase Logística (auto-ingesta web)")
st.caption("Crawling legal (robots.txt), reglas por peso/volumen, diccionario vivo y baseline ML.")

params = load_params()
sources, max_pages, delay = load_sources()

# Inicializar base de datos y cargar diccionario
db.init_db()
if "dict_df" not in st.session_state:
    st.session_state.dict_df = db.load_dictionary()

if "scheduler_thread" not in st.session_state:
    st.session_state.scheduler_thread = None
if "scheduler_stop_event" not in st.session_state:
    st.session_state.scheduler_stop_event = threading.Event()
if "scheduler_interval" not in st.session_state:
    st.session_state.scheduler_interval = 1

def merge_rows(rows):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    merged = pd.concat([st.session_state.dict_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["hash_row"], keep="first")
    st.session_state.dict_df = merged

def start_scheduler():
    if st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive():
        return
    st.session_state.scheduler_stop_event = threading.Event()
    t = threading.Thread(
        target=schedule_crawl,
        args=(st.session_state.scheduler_interval, merge_rows, st.session_state.scheduler_stop_event),
        daemon=True,
    )
    st.session_state.scheduler_thread = t
    t.start()

def stop_scheduler():
    if st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive():
        st.session_state.scheduler_stop_event.set()
        st.session_state.scheduler_thread = None

# Sidebar: parámetros y fuentes
with st.sidebar:
    st.header("⚙️ Parámetros")
    params["divisor_volumetrico"] = st.number_input("Divisor volumétrico (cm)", value=int(params["divisor_volumetrico"]), step=100)
    thr_df = pd.DataFrame(params["clases_por_peso"])
    st.markdown("**Umbrales (kg_min → clase)**")
    thr_df = st.data_editor(thr_df, num_rows="dynamic", use_container_width=True, key="thr")
    params["clases_por_peso"] = thr_df.sort_values("kg_min", ascending=True).to_dict(orient="records")
    if st.button("💾 Guardar parámetros", use_container_width=True):
        save_params(params)
        st.success("Parámetros guardados.")

    st.header("🌐 Fuentes (dominios)")
    st.caption("La app explorará estos dominios respetando robots.txt y límites.")
    src_text = st.text_area("Dominios (uno por línea)", value="\n".join(sources), height=150)
    colA, colB = st.columns(2)
    with colA:
        max_pages = st.number_input("Máx. páginas por dominio", value=int(max_pages), min_value=5, step=5)
    with colB:
        delay = st.number_input("Delay entre requests (seg)", value=float(delay), min_value=0.5, step=0.5)
    if st.button("💾 Guardar fuentes", use_container_width=True, key="save_sources"):
        new_sources = [s.strip() for s in src_text.splitlines() if s.strip()]
        save_sources(new_sources, max_pages, delay)
        st.success("Fuentes guardadas.")

st.subheader("1) 📥 Ingesta automática desde la web (crawler)")
st.write("La app recorrerá cada dominio y extraerá fichas de producto con **JSON-LD Product** cuando existan.")

st.markdown("**Programar ingesta periódica**")
st.session_state.scheduler_interval = st.number_input(
    "Frecuencia de ejecución (horas)",
    min_value=1,
    value=int(st.session_state.scheduler_interval),
    key="scheduler_interval",
)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(
        "▶️ Iniciar scheduler",
        use_container_width=True,
        disabled=st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive(),
        key="start_sched",
    ):
        start_scheduler()
with col2:
    if st.button(
        "⏹️ Detener scheduler",
        use_container_width=True,
        disabled=not (st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive()),
        key="stop_sched",
    ):
        stop_scheduler()
with col3:
    if st.button(
        "🔄 Reiniciar scheduler",
        use_container_width=True,
        disabled=not (st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive()),
        key="restart_sched",
    ):
        stop_scheduler()
        start_scheduler()

if st.button("🚀 Ejecutar ingesta web ahora", use_container_width=True):
    sources, max_pages, delay = load_sources()
    progress = st.progress(0.0, text="Iniciando...")
    all_rows = []
    for i, domain in enumerate(sources):
        progress.progress((i)/max(1, len(sources)), text=f"Crawling: {domain}")
        try:
            rows = crawl_domain(domain, params["clases_por_peso"], params["divisor_volumetrico"], max_pages=max_pages, delay=delay)
            all_rows.extend(rows)
        except Exception as e:
            st.warning(f"Error en {domain}: {e}")
    progress.progress(1.0, text="Completado.")
    if all_rows:
        new_df = pd.DataFrame(all_rows)
        # de-dup por hash_row
        merged = pd.concat([st.session_state.dict_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["hash_row"], keep="first")
        st.session_state.dict_df = merged
        for _, row in new_df.iterrows():
            db.upsert_product(row.to_dict())
        st.success(f"Ingesta completa. Nuevos registros: {len(new_df)} | Total en diccionario: {len(st.session_state.dict_df)}")
    else:
        st.info("No se encontraron productos (revisa dominios, robots.txt o aumenta el límite de páginas).")

st.markdown("Vista del diccionario (primeros 200):")
st.dataframe(st.session_state.dict_df.head(200), use_container_width=True, height=350)

st.subheader("2) 🔎 Búsqueda por nombre + validación humana")
q = st.text_input("Buscar producto por nombre")
if q:
    names = st.session_state.dict_df["product_name"].fillna("").tolist()
    # fuzzy top-10
    matches = process.extract(q, names, scorer=fuzz.WRatio, limit=10)
    idxs = []
    for m, score, pos in matches:
        # buscar índice de la fila con ese nombre (primera coincidencia)
        idx = st.session_state.dict_df.index[st.session_state.dict_df["product_name"] == m].tolist()
        if idx:
            idxs.append(idx[0])
    res = st.session_state.dict_df.loc[idxs] if idxs else pd.DataFrame()
    st.caption(f"Top {len(res)} resultados (fuzzy):")
    st.dataframe(res, use_container_width=True)

    if not res.empty:
        sel = st.selectbox("Selecciona fila para editar", res.index.tolist())
        if sel is not None:
            edit_row = st.session_state.dict_df.loc[sel].to_dict()
            st.write("**Detalle seleccionado**")
            st.json(edit_row)
            # editar clase
            classes = [c["clase"] for c in params["clases_por_peso"]]
            new_class = st.selectbox("Editar clase", classes, index=max(0, classes.index(edit_row.get("clase_logistica")) if edit_row.get("clase_logistica") in classes else 0))
            if st.button("💾 Guardar cambios en fila seleccionada"):
                st.session_state.dict_df.at[sel, "clase_logistica"] = new_class
                db.upsert_product(st.session_state.dict_df.loc[sel].to_dict())
                st.success("Actualizado.")

st.subheader("3) ⬇️ Exportar diccionario")
export_df = db.load_dictionary()
csv_data = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", csv_data, file_name="diccionario_logistica.csv", mime="text/csv")

st.divider()
st.subheader("4) 🔬 (Opcional) Entrenar baseline ML (texto → clase)")
ml_df = st.session_state.dict_df.dropna(subset=["product_name","clase_logistica"])
enough = ml_df["clase_logistica"].nunique() >= 2 and len(ml_df) >= 50
st.caption("Usa los datos del diccionario (con tus correcciones). Se requiere ≥50 filas y ≥2 clases.")
if st.button("Entrenar modelo", disabled=not enough):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            ml_df["product_name"], ml_df["clase_logistica"], test_size=0.2, random_state=42, stratify=ml_df["clase_logistica"]
        )
        vect = TfidfVectorizer(min_df=2, ngram_range=(1,2))
        Xtr = vect.fit_transform(X_train); Xte = vect.transform(X_test)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))
        st.text(classification_report(y_test, y_pred))
        # Guardar artefactos en sesión (sencillo)
        st.session_state.ml_vect = vect
        st.session_state.ml_clf = clf
        st.success("Modelo entrenado en memoria.")
    except Exception as e:
        st.error(f"Error entrenando modelo: {e}")

test_text = st.text_input("Probar predicción ML (nombre de producto)")
if test_text and "ml_vect" in st.session_state:
    pred = st.session_state.ml_clf.predict(st.session_state.ml_vect.transform([test_text]))[0]
    st.info(f"Predicción de clase: **{pred}**")
elif test_text:
    st.warning("Entrena el modelo primero.")
