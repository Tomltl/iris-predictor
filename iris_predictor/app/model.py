import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import os

DB_PATH = "data/iris.db"
SQL_SCRIPT_PATH = "data/db.sql"

def init_database():
    """Crée la base SQLite et insère les données via db.sql"""
    if not os.path.exists(SQL_SCRIPT_PATH):
        raise FileNotFoundError("Le fichier db.sql est manquant. Générez-le d'abord.")

    conn = sqlite3.connect(DB_PATH)
    with open(SQL_SCRIPT_PATH, "r") as f:
        sql_script = f.read()
        conn.executescript(sql_script)
    conn.commit()
    conn.close()
    print("✅ Base de données initialisée à partir de db.sql")

def load_data_from_db():
    """Charge les données depuis SQLite avec pandas"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM iris", conn)
    conn.close()
    return df

def train_and_log_model():
    df = load_data_from_db()

    # Ici, on veut prédire la longueur du sépale à partir de la largeur du sépale
    X = df[["sepal_width"]]
    y = df["sepal_length"]

    model = LinearRegression()
    model.fit(X, y)

    # Log avec MLflow
    mlflow.set_experiment("Iris Predictor")
    with mlflow.start_run():
        mlflow.log_param("input_feature", "sepal_width")
        mlflow.log_param("target", "sepal_length")
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("✅ Modèle entraîné et loggé avec MLflow")

if __name__ == "__main__":
    init_database()
    train_and_log_model()
