import os
import json
import time
import schedule
from typing import Callable
from web_ingestor import crawl_web

DEFAULT_PARAMS_PATH = "parametros_logistica.json"

def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"divisor_volumetrico": 5000, "clases_por_peso": []}

def crawl_with_query(query: str, max_pages: int, delay: float) -> list:
    params = load_params()
    try:
        return crawl_web(
            query,
            params.get("clases_por_peso", []),
            params.get("divisor_volumetrico", 5000),
            max_pages=max_pages,
            delay=delay,
        )
    except Exception:
        return []


def schedule_crawl(
    interval_hours: int = 24,
    query: str = "producto",
    max_pages: int = 25,
    delay: float = 1.0,
    update_fn: Callable[[list], None] | None = None,
    stop_event=None,
):
    schedule.clear()

    def job():
        rows = crawl_with_query(query, max_pages, delay)
        if update_fn:
            update_fn(rows)

    schedule.every(interval_hours).hours.do(job)
    job()
    while not (stop_event and stop_event.is_set()):
        schedule.run_pending()
        time.sleep(1)
