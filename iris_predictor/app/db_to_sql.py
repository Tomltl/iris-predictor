import pandas as pd
import os

CSV_PATH = "data/iris.csv"
SQL_PATH = "data/db.sql"
TABLE_NAME = "iris"

def convert_csv_to_sql(csv_path, sql_path, table_name):
    df = pd.read_csv(csv_path)
    columns = df.columns

    # Génération de la requête CREATE TABLE
    sql = f"DROP TABLE IF EXISTS {table_name};\n"
    sql += f"CREATE TABLE {table_name} (\n"
    for col in columns:
        dtype = "REAL" if df[col].dtype in ["float64", "int64"] else "TEXT"
        sql += f"    {col} {dtype},\n"
    sql = sql.rstrip(",\n") + "\n);\n\n"

    # Génération des INSERT INTO
    for _, row in df.iterrows():
        values = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in row])
        sql += f"INSERT INTO {table_name} VALUES ({values});\n"

    # Écriture dans le fichier
    with open(sql_path, "w") as f:
        f.write(sql)

    print(f"✅ SQL généré dans : {sql_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    convert_csv_to_sql(CSV_PATH, SQL_PATH, TABLE_NAME)
