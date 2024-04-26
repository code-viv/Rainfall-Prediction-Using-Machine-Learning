# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:49:03 2024

@author: Prerna
"""

from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np

model = joblib.load("xgb_model")
sc = pickle.load(open("scaler.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/pred', methods = ["POST"]) 
def prediction():
    
    date = request.form["date"]
    location = request.form["location"]    
    mintemp = request.form["mintemp"]
    maxtemp = (request.form["maxtemp"])
    rainfall = request.form["rainfall"]
    evaporation = request.form["evaporation"]
    windgustdir = request.form["windgustdir"]
    windgustspeed = request.form["windgustspeed"]
    winddir9am = request.form["winddir9am"]
    winddir3pm = request.form["winddir3pm"]
    windspeed9am = request.form["windspeed9am"]
    windspeed3pm = request.form["windspeed3pm"]
    humidity9am = request.form["humidity9am"]
    humidity3pm = request.form["humidity3pm"]
    pressure9am = request.form["pressure9am"]
    pressure3pm = request.form["pressure3pm"]
    cloud9am = request.form["cloud9am"]
    cloud3pm = request.form["cloud3pm"]
    temp9am = request.form["temp9am"]
    temp3pm = request.form["temp3pm"]
    sunshine = request.form["sunshine"]
    raintoday = request.form["raintoday"]
    
    dated = (float(date[5:7])-1)
    
    v = model.predict((sc.transform([[dated, float(location), float(mintemp), float(maxtemp), float(rainfall), float(evaporation), float(sunshine), float(windgustspeed), float(windspeed9am), float(windspeed3pm), float(humidity9am), float(humidity3pm), float(pressure9am), float(pressure3pm), float(cloud9am), float(cloud3pm), float(temp9am), float(temp3pm), float(windgustdir), float(winddir9am), float(winddir3pm), float(raintoday)]])))
    
    if v[0] == 0:
        return render_template("rainy_no.html")
    else:
        return render_template("rainy_yes.html")
    
    
   # return render_template("rainy_yes.html", d=(float(date[5:7])-1), mint=float(raintoday))

                        
if __name__ == "__main__":
    app.run(debug=False)
    
