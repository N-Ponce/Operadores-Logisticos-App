\
import streamlit as st
import pandas as pd
import json, os, time, io
import numpy as np
import threading
from web_ingestor import crawl_web
from scheduler import schedule_crawl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from rapidfuzz import process, fuzz
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


def save_params(params, path=DEFAULT_PARAMS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


st.set_page_config(page_title="Clasificador Logístico Ripley", page_icon="📦", layout="wide")
st.title("📦 Clasificador de Clase Logística (auto-ingesta web)")
st.caption("Crawling legal (robots.txt), reglas por peso/volumen, diccionario vivo y baseline ML.")

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
        args=(
            st.session_state.scheduler_interval,
            st.session_state.search_query,
            int(st.session_state.max_pages),
            float(st.session_state.delay),
            merge_rows,
            st.session_state.scheduler_stop_event,
        ),
        daemon=True,
    )
    st.session_state.scheduler_thread = t
    t.start()

def stop_scheduler():
    if st.session_state.scheduler_thread and st.session_state.scheduler_thread.is_alive():
        st.session_state.scheduler_stop_event.set()
        st.session_state.scheduler_thread = None

# Sidebar: parámetros
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

st.subheader("1) 📥 Ingesta automática desde la web (crawler)")
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

st.markdown("**Programar ingesta periódica**")
st.number_input(
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
            ml_df["product_name"], ml_df["clase_logistica"], test_size=0.2,
            random_state=42, stratify=ml_df["clase_logistica"]
        )

        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        maxlen = 20
        Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen, padding="post")
        Xte = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen, padding="post")

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        model = Sequential([
            Embedding(input_dim=10000, output_dim=16, input_length=maxlen),
            GlobalAveragePooling1D(),
            Dense(len(le.classes_), activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(Xtr, y_train_enc, epochs=10, verbose=0)

        y_pred_enc = np.argmax(model.predict(Xte), axis=1)
        y_pred = le.inverse_transform(y_pred_enc)
        st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))
        st.text(classification_report(y_test, y_pred))

        st.session_state.ml_tokenizer = tokenizer
        st.session_state.ml_model = model
        st.session_state.ml_le = le
        st.session_state.ml_maxlen = maxlen
        st.success("Modelo entrenado en memoria.")
    except Exception as e:
        st.error(f"Error entrenando modelo: {e}")

test_text = st.text_input("Probar predicción ML (nombre de producto)")
if test_text and "ml_model" in st.session_state:
    seq = pad_sequences(
        st.session_state.ml_tokenizer.texts_to_sequences([test_text]),
        maxlen=st.session_state.ml_maxlen, padding="post"
    )
    pred_enc = np.argmax(st.session_state.ml_model.predict(seq), axis=1)
    pred = st.session_state.ml_le.inverse_transform(pred_enc)[0]
    st.info(f"Predicción de clase: **{pred}**")
elif test_text:
    st.warning("Entrena el modelo primero.")
