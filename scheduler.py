import os
import json
import time
import yaml
import schedule
from typing import Callable
from web_ingestor import crawl_domain

DEFAULT_PARAMS_PATH = "parametros_logistica.json"
DEFAULT_SOURCES_YML = "sources.yml"

def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"divisor_volumetrico": 5000, "clases_por_peso": []}

def load_sources(path: str = DEFAULT_SOURCES_YML):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            return y.get("sources", []), y.get("max_pages_per_domain", 25), y.get("delay_seconds", 1.0)
    return [], 25, 1.0

def crawl_all_sources() -> list:
    params = load_params()
    sources, max_pages, delay = load_sources()
    all_rows = []
    for domain in sources:
        try:
            rows = crawl_domain(
                domain,
                params.get("clases_por_peso", []),
                params.get("divisor_volumetrico", 5000),
                max_pages=max_pages,
                delay=delay,
            )
            all_rows.extend(rows)
        except Exception:
            continue
    return all_rows

def schedule_crawl(interval_hours: int = 24, update_fn: Callable[[list], None] | None = None, stop_event=None):
    schedule.clear()
    def job():
        rows = crawl_all_sources()
        if update_fn:
            update_fn(rows)
    schedule.every(interval_hours).hours.do(job)
    job()
    while not (stop_event and stop_event.is_set()):
        schedule.run_pending()
        time.sleep(1)
