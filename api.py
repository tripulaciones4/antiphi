from flask import Flask, request
from utils.utils import app_prediction
import mysql.connector

app = Flask(__name__)

app.config["DEBUG"] = True


@app.route('/', methods = ['GET'])
def get_info():
    '''
    This function returns the status of a url saved in our database (legitime or phishig).
    In case it is not in the database, it will use a prediction model to say the status and will save it.
    '''

    connection = mysql.connector.connect(
    username = 'Desafio2109sep',
    password = 'Desafio21Tripulaciones09',
    host = 'desafio-tripulaciones.c7f2y5gspcuf.eu-west-3.rds.amazonaws.com',
    port = 3306,
    database = 'desafio_tripulaciones_db'
    )

    cursor = connection.cursor()

    url = request.args['url']

    if url is None:
        return "Missing values, the data connot be provided"

    else:
        try:
            query_get = "SELECT * FROM web_table WHERE url LIKE %s;"
            cursor.execute(query_get, (url,))
            url, result = cursor.fetchone()

        except:
            print('Not in the DB')
            result = app_prediction(url, '/home/desafiotripulaciones4/antiphi/models/One_Hot_Encoder', '/home/desafiotripulaciones4/antiphi/models/web_predictor.model')
            url_predicted = [url, result]
            query_save = 'INSERT INTO web_table (url, status) VALUES (%s,%s);'
            cursor.execute(query_save, url_predicted)
            connection.commit()
            connection.close()
            return {'url':url,'result': result}

        connection.commit()
        connection.close()
        return {'url':url,'result': result}

if __name__ == '__main__':
    app.run()
