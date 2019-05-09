
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en')

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
#nltk.download('wordnet')

class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight


    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix

		        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        return g_norm


    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keywords_list=[]
        for i, (key, value) in enumerate(node_weight.items()):
            keywords_list.append(key)
            if ++i == number:
               break
        return keywords_list
    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        self.node_weight = node_weight

## 2. Get Top 100 Keywords from DB

def get_top_keywords_from_db():
    conn = psycopg2.connect(host="13.234.140.137",database="ingenmasterdb", user="ingenworks", password="Ingen@123")
    cursor = conn.cursor()
    cursor.execute("""select row_labels from public."ingenspark_Keyword_normalized_scores" order by "sum_of_adjusted_score" limit 100""")
    keywords= cursor.fetchall()
    db_keywords = [x[0] for x in keywords]
    #print(db_keywords)
    return db_keywords

# 3. Compare Keywords from DB and Project description
app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/getkeywords')
@cross_origin(supports_credentials=True)
def get_tasks():
    project_desc = request.args.get('projectdescription')
    tr4w = TextRank4Keyword()
    tr4w.analyze(project_desc, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=True)
    #tr4w.get_keywords(10)
    db_keywords = get_top_keywords_from_db()
    finalKeywords = list(set(tr4w.get_keywords(10)).intersection(db_keywords))
    #print(finalKeywords)
    res= json.dumps(finalKeywords)
    return res

if __name__ == '__main__':

    app.run(host='0.0.0.0')




