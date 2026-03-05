from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# load the strained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tf_idf.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)[0]  # Use the prediction result directly
    return render_template('index.html', message=message, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
