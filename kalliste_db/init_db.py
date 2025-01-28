"""Initialize the Kalliste SQLite database."""
import sqlite3
import os
from pathlib import Path
from kalliste.config import KALLISTE_DB_PATH


SCHEMA_PATH = Path(__file__).parent / "schema.sql"

def init_db():
    """Initialize SQLite database using schema."""
    db_dir = os.path.dirname(KALLISTE_DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    if os.path.exists(KALLISTE_DB_PATH):
        os.remove(KALLISTE_DB_PATH)
    
    with sqlite3.connect(KALLISTE_DB_PATH) as conn:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        print(f"Database initialized at {KALLISTE_DB_PATH}")

if __name__ == "__main__":
    init_db()