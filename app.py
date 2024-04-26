import pygal
from flask import Flask, render_template, request,url_for,redirect
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


app = Flask(__name__)


def preprocessing():
    global df, x_test, x_train, y_test, y_train
    df = pd.read_csv(r'dataset\water_potability.csv')

    df["ph"].fillna(method='bfill', limit=30, inplace=True)
    df["Sulfate"].fillna(method='ffill', limit=30, inplace=True)
    
    df["Trihalomethanes"].fillna(method='bfill', limit=30, inplace=True)

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return x_train, x_test, y_train, y_test


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/training', methods=['GET','POST'])
def training():

    x_train, x_test, y_train, y_test = preprocessing()


    if request.method == "POST":

        model= int(request.form['algo'])
        if model==1:
            x_train, x_test, y_train, y_test = preprocessing()
            rf = RandomForestClassifier(random_state=10, criterion='gini')
            rf.fit(x_train, y_train)
            rfc = rf.predict(x_test)
            global rfa,gbcrf,abcrf,nba,aba,cba
            rfa = accuracy_score(y_test, rfc)
            rfa = 'Accuracy of RandomForest:' + str(rfa)
            return render_template('training.html',msg=rfa,rfa=rfa)
        elif model == 2:
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            y_pred = gnb.predict(x_test)
            nba = accuracy_score(y_test, y_pred)
            nba = 'Accuracy of Naive Bayes: ' + str(nba)
            return render_template('training.html',msg=nba,nba=nba)
        elif model==3:
            return render_template('training.html',msg="Please select a model")
    return render_template('training.html')
d={}
d1={}

@app.route('/detection', methods=['GET','POST'])
def detection():
    if request.method == "POST":
        ph=request.form['ph']
        print(ph)
        Hardness = request.form["Hardness"]
        print(Hardness)
        Solids = request.form["Solids"]
        print(Solids)
        Chloramines = request.form["Chloramines"]
        print(Chloramines)
        Sulfate = request.form["Sulfate"]
        print(Sulfate)
        Conductivity = request.form["Conductivity"]
        print(Conductivity)
        Organic_carbon = request.form["Organic_carbon"]
        print(Organic_carbon)
        Trihalomethanes = request.form["Trihalomethanes"]
        print(Trihalomethanes)
        Turbidity = request.form["Turbidity"]
        print(Turbidity)
        mna=[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
        x_train, x_test, y_train, y_test = preprocessing()
        model= RandomForestClassifier(random_state=10)
        model.fit(x_train, y_train)
        output=model.predict([mna])
        print(output)
        if output==1:
            val='<b><span style = color:black;>The Water Quality  <span style = color:red;>is Pure</span></span></b>'
        elif output==0:
            val='<b><span style = color:black;>The Water Quality<span style = color:red;>is Not Pure</span></span></b>'
        return render_template('detection.html',msg=val)
    return render_template('detection.html')

@app.route("/graph",methods=['GET','POST'])
def graph():
    return render_template('graph.html')




if __name__ == '__main__':
    app.run(debug=True)
