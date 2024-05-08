from flask import Flask, render_template, request, jsonify
from gensim.models import FastText
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.tokenize import word_tokenize

app = Flask(__name__)

max_length = 50
fasttext_model = FastText.load('./models/model-pretraineed.50.fasttext')
model = load_model("./models/model-ft-90.keras")

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    texts = data['query']

    sentences = word_tokenize(texts)
    # print(sentences)

    # Tokenize and convert text to FastText embeddings
    embedded_texts = []
    for sentence in sentences:
        embedded_text = [fasttext_model.wv[word] for word in sentence if word in fasttext_model.wv]
        if len(embedded_text) > 0:
            embedded_texts.append(embedded_text)
    
    # Pad sequences to make them of equal length
    padded_texts = pad_sequences(embedded_texts, maxlen=max_length, padding='post')

    result = model(padded_texts);
    return jsonify({'status': 'success', 'data': sentences})

@app.route('/')
def index():    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
