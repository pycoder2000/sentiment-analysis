import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)
vect = pickle.load(open("vectorizer.pickle",'rb'))
model = tf.keras.models.load_model('model.h5')

def preprocess(text, stem=False):
  text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
  stop_words = stopwords.words('english')
  stemmer = SnowballStemmer('english')
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inp = [x for x in request.form.values()]
    inp = preprocess(inp)
    inp = vect.transform([inp]).toarray().reshape(1,1,2500)
    output = model.predict(inp)[0]

    return render_template('index.html', prediction_text='{}'.format(output[0]), anchor="services")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)