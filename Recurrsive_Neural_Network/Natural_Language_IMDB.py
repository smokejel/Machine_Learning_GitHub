from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

# now time to make a prediction

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred)
  print(result[0])

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

# Importing and Proprocessing Data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

# Model Definition
model = tf.keras.Sequential([tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")])

model.summary()

# Compile and train model
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# Making Predictions
word_index = imdb.get_word_index()

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}

print(decode_integers(encoded))

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)
