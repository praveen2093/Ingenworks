import json
from flask import Flask, jsonify, request
from flask import request,abort
from flask_cors import CORS, cross_origin
import psycopg2

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/addkeywords', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_task():
    jdata = request.get_json()
    for item in jdata['keywords']:
        userid= item['userid']
        keyword= item['keyword']
        action= item['action']
        res="successfully logged"
        conn = psycopg2.connect(host="13.234.140.137",database="ingenmasterdb", user="ingenworks", password="Ingen@123")
        cursor = conn.cursor()
        query = "insert into ingenspark_keywords_input(row_labels,action) VALUES ('"+keyword+"','"+action+"')"
        cursor.execute(query)
        conn.commit()
    return jsonify({'status': res}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)
