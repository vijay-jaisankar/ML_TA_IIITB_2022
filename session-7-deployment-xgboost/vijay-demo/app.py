from flask import Flask, render_template, request
import pickle
import pandas as pd 

app = Flask(__name__)

mapping_dict = {'apple': 0,
 'banana': 1,
 'blackgram': 2,
 'chickpea': 3,
 'coconut': 4,
 'coffee': 5,
 'cotton': 6,
 'grapes': 7,
 'jute': 8,
 'kidneybeans': 9,
 'lentil': 10,
 'maize': 11,
 'mango': 12,
 'mothbeans': 13,
 'mungbean': 14,
 'muskmelon': 15,
 'orange': 16,
 'papaya': 17,
 'pigeonpeas': 18,
 'pomegranate': 19,
 'rice': 20,
 'watermelon': 21
}

def reverse_mapping(x):
    for k in mapping_dict:
        if mapping_dict[k] == x:
            return k 
    return "No Crop"


with open("saved_model.sav", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods = ["GET", "POST"])
def book():
    if request.method == 'POST':
        # Get Book Summary
        N = request.form["N"]
        P = request.form["P"]
        K = request.form["K"]
        temperature = request.form["temperature"]
        humidity = request.form["humidity"]
        ph = request.form["ph"]
        rainfall = request.form["rainfall"]

        df = pd.DataFrame({
            "N" : float(N),
            "P" : float(P), 
            "K" : float(K),
            "temperature" : float(temperature),
            "humidity" : float(humidity),
            "ph" : float(ph), 
            "rainfall" : float(rainfall)
        }, index = [0])


        pred = model.predict(df)
        label = reverse_mapping(pred[0])

        return render_template("index.html", label = label)
    
    else:
        return render_template("index.html", label = None)