
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

@app.route('/recommendations')
@cross_origin(supports_credentials=True)
def get_tasks():
    key_words1 = request.args.get('key_words')
    matched=pd.DataFrame(get_matches(key_words1,choices['key_words']))
    matched.columns = ["key_words", "Score1", "Score2"]
    data = matched.drop(["Score1","Score2"],axis=1)
    df = pd.merge(data, choices, on='key_words')
    df1=df['project_id']
    #print(df1)
    df2=df1.to_string(index=False).split('\n')
    df2 = [x.replace(' ', '') for x in df2]
    #print(df2)
    df2=df1.to_string(index=False).split('\n')
    df2 = [x.replace(' ', '') for x in df2]
    #print(df2)
    def unique(list1):
      # intilize a null list
        unique_list = []
      # traverse for all elements
        for x in list1:
          # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        for x in unique_list:
            print(x)
    #list1=unique(df2)
    df3=set(df2)
    #print(df3)
    class SetEncoder(json.JSONEncoder):
         def default(self, obj):
           if isinstance(obj, set):
               return list(obj)
           if isinstance(obj, Something):
               return 'CustomSomethingRepresentation'
               return json.JSONEncoder.default(self, obj)
    res= json.dumps({'recommedations':df3}, cls=SetEncoder)
    return res


if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5004)


