import train_data
import json
import numpy as np
from flask import request
from flask import Flask

app = Flask(__name__)

'''
Tek parametre data olacak post requestte. datanın içinde stringe dönüştürülmüş json verisi olacak.
port request flaskın ayağa kalktığı portun /start route'una atılacak.

Flask komutları
** main.py'nin bulunduğu dizine gel alttaki komutları gir
** flask senin local ip'nin 5000. portunda ayağa kalkar. local ip'yi görmek için komut satırına "ipconfig" yazıp ipv4 adresinde yazan yeri al.
1- SET FLASK_APP=main
2-flask run --host=0.0.0.0

'''



@app.route("/start", methods=['POST'])
def start():
    # result = request['halo']
    data = request.form['data']
    # print(request.json)
    # return str(result)
    data = json.loads(data)
    result = train_data.start_task(data)
    # result = "Logaritmic Mean squared error: {0}".format(train_data.rmsle(test_labels, predictions))
    return str(result)


@app.route("/")
def hello():
    return 'main'


if __name__ == '__main__':
    app.run()