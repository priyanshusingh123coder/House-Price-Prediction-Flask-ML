from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model (VERY IMPORTANT)
model = pickle.load(open("model/house_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    area = int(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    location = int(request.form["location"])

    # DEBUG: print inputs (check in terminal)
    print("Inputs:", area, bedrooms, bathrooms, location)

    # Predict price
    prediction = model.predict([[area, bedrooms, bathrooms, location]])
    price = int(prediction[0])

    return render_template("index.html", result=price)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
