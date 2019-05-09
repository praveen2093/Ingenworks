
from sqlalchemy import create_engine
import io
#import psycopg2 as pg
import pandas.io.sql as psql
import pandas as pd
import json
from flask import Flask, jsonify, request
from flask import request,abort
from flask_cors import CORS, cross_origin
import psycopg2

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/KeywordScoring')
@cross_origin(supports_credentials=True)
def create_task():
    res="successfully added"
    connection = psycopg2.connect(host="13.234.140.137",database="ingenmasterdb",user="ingenworks",
password="Ingen@123")
    df = pd.read_sql_query('select * from ingenspark_keywords_input',con=connection)
    df = df.rename(columns={'user_id':'created_user_id'})
    df = df.rename(columns={'count_of_score':'score'})
    df['Today']= pd.Timestamp("today").strftime("%Y-%m-%d")
    df['Today'] = pd.to_datetime(df['Today']).dt.date
    df['Date_of_action'] =pd.to_datetime(df['created_dttm']).dt.date
    df['Recency']=(df['Today'] - df['Date_of_action']).dt.days
    df['Recency_in_Quarters']=df['Recency']/90
    df['Adjustment_Factor']=df['Recency_in_Quarters']*0.1
    df['sum_of_adjusted_score']=(abs(df['score'])-df['Adjustment_Factor'])*(df['score'])
    df['normalized_score']=(df['Adjustment_Factor'])/(df['score'])
    df=df.drop(['Recency','Adjustment_Factor','Recency_in_Quarters','Today','Date_of_action','Recency_in_Quarters','Adjustment_Factor','project_id','action'], axis=1)
    df.index = df.index + 1
    df.index.name = 'ingenspark_keyword_scores_id'
    df['created_dttm'] = (df['created_dttm']).apply(lambda d: pd.to_datetime(str(d)))
    engine = create_engine('postgresql://ingenworks:Ingen@123@13.234.140.137:5432/ingenmasterdb', echo=False)
    cursor = connection.cursor()
    cursor.execute("TRUNCATE TABLE ingenspark_keyword_scores RESTART IDENTITY")
    connection.commit()
    df.to_sql(name='ingenspark_keyword_scores',con=engine,if_exists='append',index=False)
    return jsonify({'status': res}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5003)


