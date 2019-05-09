import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import psycopg2
import pandas as pd
import json
import nltk
from pandas import DataFrame
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
import psycopg2
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/recommendations')
@cross_origin(supports_credentials=True)
def get_tasks():
    project_nm = request.args.get('project_desc').split(',')[0]
    #project_nm = "saline"
    print(project_nm)
    num_of_recommendations = int(request.args.get('limit'))
    #num_of_recommendations = 3
    print(num_of_recommendations)
    connection = psycopg2.connect(
        host="13.234.140.137",
        database="ingenmasterdb",
        user="ingenworks",
        password="Ingen@123"
    )
    tf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0,
        stop_words='english'
    )

    tfidf_matrix = tf.fit_transform(ds['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['project_nm'][i]) for i in similar_indices]

        # First item is the item itself, so remove it.
        # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
        results[row['project_nm']] = similar_items[1:]

    recs = results[project_nm][:num_of_recommendations]
    recommendations = []
    for rec in recs:
        resp = ds.loc[ds['project_nm'] == rec[1]]['project_id'].tolist()[0]
        recommendations.append(resp)

    return jsonify(recommendations=recommendations)


if __name__ == "__main__":

    app.run(host='0.0.0.0',port=5004)

