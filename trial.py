import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers
from keras.layers import TextVectorization
import pickle

# Load the tensorflow model
model = load_model('model.h5', compile = False)


model.compile(
optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["binary_accuracy"]
)


# Load the vectorizer layer
from_disk = pickle.load(open("tv_layer.pkl", "rb"))

print(from_disk['config'])
# vectorizer = TextVectorization.from_config(config)


# Configure the vectorizer
vectorizer = TextVectorization(
    # name = 'text_vectorization', 
    # trainable = True, 
    dtype = 'string',
    max_tokens = from_disk['config']['max_tokens'],
    output_mode = "tf-idf",
    ngrams = 1,
    vocabulary_size = from_disk['config']['vocabulary_size']
    )

# data = pd.read_csv('train.csv')
# data = data.drop(['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
# vectorizer.adapt(data)

# Set the weights of the vectorizer
vectorizer.set_weights(from_disk['weights'])
print(model.summary())

dictionary = {'comment_text': ["I hate your product. I hate you. I hope you rot in hell. You are a no good nigger"]}
df = pd.DataFrame.from_dict(dictionary)
query_df = vectorizer(df)
print(query_df)

test_df = pd.read_csv('test.csv')
test_df = test_df.sample(n=5000, axis = 0)
test_df = vectorizer(test_df["comment_text"])
predictions = model.predict(query_df)

def answer(array):
    ll = []
    for i in array:
        # What are the indexes with numbers greater than 0.5
        column = np.where(i > 0.5)

        # Convert the array of indexes to lists and append it to a bigger list
        ll.append(list(column[0]))
        # print(column)
    return ll

# Map the function to the numpy array
f = lambda x: answer(x)
predictions_1 = f(predictions)

labels = ['toxic', 'severe_toxic',  'obscene', 'threat', 'insult', 'identity_hate', 'clean']

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
    
prediction_label = labelling(predictions_1)
print(prediction_label[:20])
#prediction = model.predict(query_df)
# return jsonify({'Prediction': list(query_df) })
#print({'Prediction': list(prediction) })

