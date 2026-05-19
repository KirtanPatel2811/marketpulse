"""
db_migration.py
Adds missing sentiment probability columns to news_articles table.
Safe to run multiple times — uses ALTER TABLE IF NOT EXISTS pattern.
"""
import sqlite3
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from config import CACHE_DB_PATH

conn = sqlite3.connect(CACHE_DB_PATH)
cursor = conn.cursor()

new_cols = [
    ("sentiment_confidence", "REAL"),
    ("sentiment_positive",   "REAL"),
    ("sentiment_negative",   "REAL"),
    ("sentiment_neutral",    "REAL"),
]

for col, dtype in new_cols:
    try:
        cursor.execute(f"ALTER TABLE news_articles ADD COLUMN {col} {dtype}")
        print(f"Added column: {col}")
    except sqlite3.OperationalError:
        print(f"Column already exists: {col}")

conn.commit()
conn.close()
print("Migration complete.")