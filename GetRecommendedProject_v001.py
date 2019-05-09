
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import io
import numpy as np
import pandas.io.sql as psql
import json

connection = psycopg2.connect(host="13.234.140.137",database="ingenmasterdb",user="ingenworks",
password="Ingen@123")
metadata = pd.read_sql_query('select * from project',con=connection)
filtered_df = metadata[metadata['key_words'].notnull()]
#filtered_df
choices=filtered_df[['project_id','description','key_words']]
#choices.head(3)

def get_matches(query,choices,limit=10,cut=65):
    results=process.extractBests(query,choices,score_cutoff=cut,limit=limit)
    return results

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/getkeywords')
@cross_origin(supports_credentials=True)
def get_tasks():
    key_words1 = request.args.get('key_words')
    projectdesc_keywords = get_matches(key_words1,choices['key_words'])
    res= json.dumps(projectdesc_keywords)
    return res

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5009)
