from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.sklearn
import os
import numpy as np

app = Flask(__name__)
CORS(app)

def load_latest_model():
    mlruns_path = "/app/mlruns/0"
    runs = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d))]
    if not runs:
        raise Exception("Aucun modèle trouvé dans mlruns/0.")
    latest_run = sorted(runs, reverse=True)[0]
    model_path = os.path.join(mlruns_path, latest_run, "artifacts", "model")
    print(f"✅ Modèle chargé depuis : {model_path}")
    return mlflow.sklearn.load_model(model_path)

model = load_latest_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "sepal_width" not in data:
        return jsonify({"error": "Le champ 'sepal_width' est requis"}), 400
    
    try:
        sepal_width = float(data["sepal_width"])
    except ValueError:
        return jsonify({"error": "Valeur de 'sepal_width' invalide"}), 400

    prediction = model.predict(np.array([[sepal_width]]))[0]
    return jsonify({"sepal_length_prediction": round(float(prediction), 2)})

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=8000, debug=True)
