from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model & vectorizer
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# Load dataset to calculate accuracy
data = pd.read_csv(os.path.join(BASE_DIR, "fake_news.csv"))
X = vectorizer.transform(data["text"])
y = data["label"]

accuracy = round(accuracy_score(y, model.predict(X)) * 100, 2)

# Store last 5 predictions
history = []

@app.route("/predict", methods=["POST"])
def predict():
    global history
    text = request.json.get("text", "")
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]

    result = "FAKE NEWS ❌" if pred == 1 else "REAL NEWS ✅"

    history.insert(0, {"text": text[:60] + "...", "result": result})
    history = history[:5]

    return jsonify({"result": result})

@app.route("/accuracy", methods=["GET"])
def get_accuracy():
    return jsonify({"accuracy": accuracy})

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True)
