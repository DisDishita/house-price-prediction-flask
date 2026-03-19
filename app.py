from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sqft = float(data["sqft"])

    prediction = model.predict(np.array([[sqft]]))[0]

    return jsonify({
        "price": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run()