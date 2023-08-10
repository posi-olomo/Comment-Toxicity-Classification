import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
import pickle

# Create flask app
app = Flask(__name__)

# Load the tensorflow model
model = load_model('model.h5', compile = False)
model.compile(
optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["binary_accuracy"]
)

# Load the vectorizer layer
from_disk = pickle.load(open("tv_layer.pkl", "rb"))

# Configure the vectorizer
vectorizer = layers.TextVectorization(
    dtype = 'string',
    max_tokens = from_disk['config']['max_tokens'],
    output_mode = "tf-idf",
    ngrams = 1,
    vocabulary_size = from_disk['config']['vocabulary_size']
    )

# Adapting it using dummy data
# vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))

# Set the weights of the vectorizer
vectorizer.set_weights(from_disk['weights'])

def answer(array):
    ll = []
    for i in array:
        # What are the indexes with numbers greater than 0.5
        column = np.where(i > 0.5)

        # Convert the array of indexes to lists and append it to a bigger list
        ll.append(list(column[0]))
        # print(column)
    return ll

labels = ['toxic', 'severe_toxic',  'obscene', 'threat', 'insult', 'identity_hate', 'clean']

def clean_data(text):
    # Remove hyperlinks
    res = re.sub('https\S+|www\S+|https\S+', '', text)

    # Remove special character
    res = re.sub('[^\w\s]', '', res)

    # Remove numbers
    res = re.sub('\d+', '', res)

    # Remove next line syntax
    res = re.sub('\n', '', res)
    
    return res
    

def labelling(lily):
    ll = []
    for i in lily:
        hh = []
        # If the list is empty, meaning it has no negative labels: 'clean'
        if i == []:
            ll.append(labels[6])
        else:
            # Create a list containing all thenegative labels for that entry
            for j in i:
                hh.append(labels[j])
            ll.append(hh)
    return ll

@app.route("/")
def upload():
    return render_template("upload.html")

# Create a predict method using the postman ability 'POST'
@app.route('/predict', methods = ['POST'])
def predict():
    comment_text  = str(request.form["comment"])
    comment_text = clean_data(comment_text)
    dictionary = {'comment_text': [comment_text]}
    df = pd.DataFrame.from_dict(dictionary)
    query_df = vectorizer(df)

    predictions = model.predict(query_df)

    # Map the function to the numpy array
    f = lambda x: answer(x)
    predictions_1 = f(predictions)

    prediction_label = labelling(predictions_1)

    
    return jsonify({'Prediction': prediction_label })

if __name__ == '__main__':
    app.run(host = '0.0.0.0',debug=True)
