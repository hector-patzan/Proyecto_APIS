from flask import Flask, request, jsonify
import pandas as pd 
import joblib
import numpy as np 
import datetime as dt
from termcolor import colored

app = Flask(__name__)

modelo_entrenado_para_produccion = joblib.load('housePrice_pipeline_v12022.pkl')
FEATURES = joblib.load('FEATURES.pkl')

def generateLog(message, logType):
    f = open("logData.log", "a")
    message = message + dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + ";" + '\\'
    f.write(message)
    if(logType==10):
        strcolor = 'yellow'
    elif(logType==30):
        strcolor = 'red'   
    elif(logType==90):
        strcolor = 'green'   

    print(colored(message, strcolor))
    f.close()

@app.route("/predicciones", methods=['POST'])
def predictOne():

    ###### ====== Codigo general para cualquier API ====== ###### 
    data = request.get_json()
    dataframe = pd.json_normalize(data) # Dejamos datos con el tipo original

    logStr = "0X10 - INFO - JSON transformado exitosamente -" 
    generateLog(logStr,10)


    ###### ====== Codigo para este API ====== ###### 

    ids = dataframe['PassengerId']
    dataframe = dataframe[FEATURES]


    ###### ====== LOGs sobre transformación de JSON ====== ###### 



    ###### ====== Prediccion ====== ###### 
    try:
        nomr_preds = modelo_entrenado_para_produccion.predict(dataframe)    
        outPredict = np.exp(nomr_preds)

        out = {}
        for index, item in enumerate(outPredict):
            out[str(ids[index])] = round(item, 2)

        logStr = "0X90 - Exito - Prediccion exitosa -"
        generateLog(logStr,90)
        return jsonify(out)

    except ValueError:
        logStr = "0X30 - PredicError - Se genero un error en la prediccion -"
        generateLog(logStr,30)
        return jsonify({'mensaje':logStr})

# Set FLASK_APP=app
# flask run