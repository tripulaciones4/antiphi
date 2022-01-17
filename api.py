from flask import Flask, request
import os
import mysql.connector
from utils.utils import app_prediction

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>API para desafío de tripulaciones.</p>"


@app.route('/api/v1/append_register', methods = ['GET'])
def append_register():

    '''
    Permite guardar nuevos registros.
    '''
    connection = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = 'password',
    database = 'webs_db'
    )
    cursor = connection.cursor()
    url = request.form['URL']
    status = request.form['status']
    status = status.lower()

    values_list = [url, status]

    if url is None or status is None:
        connection.commit()
        connection.close()
        return "Missing values, the data were not stored"

    else:
        cursor.execute('INSERT INTO webs (url, status) VALUES (%s,%s);', values_list)

    connection.commit()
    connection.close()
    return 'Data stored'



@app.route('/api/v1/resources/get_register', methods=['GET'])
def get_info():

    '''
    Devuelve si la web es ligítima o phishing en caso de que esté en la base de datos.
    '''
    connection = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = 'password',
    database = 'webs_db'
    )
    cursor = connection.cursor()
    url = request.form['URL']

    if url is None:
        return "Missing values, the data connot be provided"

    else:
        try:
            query_get = "SELECT * FROM webs WHERE url LIKE %s;"
            cursor.execute(query_get, (url,))
            url, result = cursor.fetchone()
            in_db = True

        except:
            print('Not in the DB')
            result = app_prediction(url, './models/One_Hot_Encoder', './models/web_predictor.model')
            url_predicted = [url, result]
            query_save = 'INSERT INTO webs (url, status) VALUES (%s,%s);'
            cursor.execute(query_save, url_predicted)
            in_db = False
            connection.commit()
            connection.close()
            return {'url':url,'result': result, 'in_database':in_db}
    connection.commit()
    connection.close()

    return {'url':url,'result': result, 'in_database':in_db}

app.run()