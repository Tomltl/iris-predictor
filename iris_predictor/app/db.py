import pandas as pd
import sqlite3  # ou `import psycopg2` si tu utilises PostgreSQL
import os
from sklearn.datasets import load_iris

DATA_PATH = "data/iris.csv"
DB_PATH = "data/iris.db"

def create_csv():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]  # clean noms de colonnes
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ CSV créé : {DATA_PATH}")

def load_csv_to_sqlite():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_csv(DATA_PATH)
    df.to_sql("iris", conn, if_exists="replace", index=False)
    print(f"✅ Données insérées dans la base SQLite : {DB_PATH}")
    conn.close()

if __name__ == "__main__":
    create_csv()
    load_csv_to_sqlite()
