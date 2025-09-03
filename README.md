# Clasificador de Clase Logística (Ripley)

App en **Streamlit** que:
- Ingresa productos automáticamente desde la web (respeta **robots.txt**, límite de páginas y demora).
- Calcula **peso facturable** (máx(real, volumétrico)) y **clase logística** por umbrales.
- Mantiene un **diccionario vivo** (editable desde UI).
- Lista para **deploy en Streamlit Community Cloud**.

## Estructura
```
.
├── app.py
├── web_ingestor.py
├── parametros_logistica.json
├── requirements.txt
└── .streamlit/config.toml
```

## Configuración rápida (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pruebas

Ejecuta los tests unitarios con:

```bash
pytest -q
```

## Deploy en Streamlit Cloud
1. Crea un repo en GitHub y sube estos archivos tal cual.
2. Ve a https://share.streamlit.io , conecta tu repo y selecciona `app.py`.
3. Dentro de la app, ajusta el término de búsqueda y usa **“🚀 Ejecutar ingesta web ahora”** para poblar el diccionario.

## Notas de cumplimiento
- La ingesta respeta **robots.txt** y aplica **delay** entre requests.
- La app solo extrae metadatos visibles públicamente (JSON-LD Product / OpenGraph).
- Se recomienda usar términos de búsqueda relacionados con sitios que tengas permiso de explorar.
