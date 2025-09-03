import sqlite3
import pandas as pd
from contextlib import closing

DB_PATH = "products.db"

COLUMNS = [
    "product_name","brand","peso_kg","largo_cm","ancho_cm","alto_cm",
    "peso_vol_kg","peso_fact_kg","clase_logistica","source_url","fetched_at","hash_row"
]


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create products table if it doesn't exist."""
    with closing(get_connection()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_name TEXT,
                brand TEXT,
                peso_kg REAL,
                largo_cm REAL,
                ancho_cm REAL,
                alto_cm REAL,
                peso_vol_kg REAL,
                peso_fact_kg REAL,
                clase_logistica TEXT,
                source_url TEXT,
                fetched_at TEXT,
                hash_row TEXT PRIMARY KEY
            )
            """
        )
        conn.commit()


def load_dictionary():
    """Return entire products table as DataFrame."""
    with closing(get_connection()) as conn:
        df = pd.read_sql_query("SELECT * FROM products", conn)
    if df.empty:
        df = pd.DataFrame(columns=COLUMNS)
    return df


def upsert_product(row):
    """Insert or update a product dictionary into the database."""
    with closing(get_connection()) as conn:
        values = [row.get(col) for col in COLUMNS]
        placeholders = ",".join(["?"] * len(COLUMNS))
        set_clause = ",".join([f"{col}=excluded.{col}" for col in COLUMNS if col != "hash_row"])
        conn.execute(
            f"""
            INSERT INTO products ({','.join(COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(hash_row) DO UPDATE SET {set_clause}
            """,
            values,
        )
        conn.commit()
