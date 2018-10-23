import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os, re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, GRU, Dense, Dot
from keras.models import Model
from keras.layers.core import *
from keras import backend as K
from keras.layers import merge

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

## The data from files are read into pandas dataframe
train = pd.read_csv("input/labeledTrainData.tsv", header = 0, delimiter = '\t')
test = pd.read_csv("input/testData.tsv", header = 0, delimiter = '\t')
test["sentiment"] = test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

## This function removes the html components from the reviews
def html2text(review):
    """Return extracted text string from provided HTML string."""
    review_text = BeautifulSoup(review, "lxml").get_text()
    if len(review_text) == 0:
        review_text = review
    review_text = re.sub(r"\<.*\>", "", review_text)
    try:
        review_text = review_text.encode('ascii', 'ignore').decode('ascii')
    except UnicodeDecodeError:
        review_text = review_text.decode("ascii", "ignore")
    return review_text

## This function removes any non alphabatic and numeric character
def letters_only(text):
    """Return input string with only letters (no punctuation, no numbers)."""
    return re.sub("[^a-zA-Z]", " ", text)

## processing review
def review_preprocess(review):
    """Preprocessing used before fitting/transforming RNN tokenizer - Html->text, remove punctuation/#s, lowercase."""
    return letters_only(str(html2text(review)).lower())

## The text is converted into a train and validation set
def get_train_data(df):
    """Extracts features (using reviews_to_features_fn), splits into train/test data, and returns
    x_train, y_train, x_test, y_test.  If no feature extraction function is provided, x_train/x_test will
    simply consist of a Series of all the reviews.
    """
    SEED = 1000
    # Shuffle data frame rows
    np.random.seed(SEED)
    df = df.iloc[np.random.permutation(len(df))]

    feature_rows = df["review"].map(review_preprocess)
    if type(feature_rows[0]) == np.ndarray:
        num_instances = len(feature_rows)
        num_features = len(feature_rows[0])
        x = np.concatenate(feature_rows.values).reshape((num_instances, num_features))
    else:
        x = feature_rows

    y = df["sentiment"]

    # Split 80/20
    test_start_index = int(df.shape[0] * .8)
    x_train = x[0:test_start_index]
    y_train = y[0:test_start_index]
    x_val = x[test_start_index:]
    y_val = y[test_start_index:]

    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = get_train_data(train)
y_test = test["sentiment"]
x_test = test["review"].map(review_preprocess)

## train, valid and test set is converted into list
train_review_list = x_train.tolist()
val_review_list = x_val.tolist()
test_review_list = x_test.tolist()
all_review_list = x_train.tolist() + x_val.tolist()

## setting parameter for Keras
np.random.seed(1000)
num_most_freq_words_to_include = 10000
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500
embedding_vector_length = 32

## Tokenization of text for generation of word embeddings
tokenizer = Tokenizer(num_words=num_most_freq_words_to_include)
tokenizer.fit_on_texts(all_review_list)

## text is converted into indexes and padding is done to get a fixed size input
train_reviews_tokenized = tokenizer.texts_to_sequences(train_review_list)
x_train = pad_sequences(train_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
val_review_tokenized = tokenizer.texts_to_sequences(val_review_list)
x_val = pad_sequences(val_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
test_review_tokenized = tokenizer.texts_to_sequences(test_review_list)
x_test = pad_sequences(test_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)


## attention layer for weightage the generated hidden vector for sentiment prediction
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    attention = Dense(1, activation='tanh')(inputs)                             # input shape = batch * time_steps * 1
    attention = Flatten()(attention)                                            # input shape = batch * time_steps
    attention = Activation('softmax')(attention)                                # input shape = batch * time_steps
    attention = RepeatVector(input_dim)(attention)                              # input shape = batch * input_dim * time_steps
    attention = Permute([2, 1])(attention)                                      # input shape = batch * time_step * input_dim
    sent_representation = merge([inputs, attention], mode='mul')                # input shape = batch * time_step * input_dim
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),               # input shape = batch * input_dim
                                 output_shape=(input_dim,))(sent_representation)
    return sent_representation


## the rnn model for sentiment analysis
def rnn_model():
    input_sequences = Input(shape=(MAX_REVIEW_LENGTH_FOR_KERAS_RNN,))
    embedding_layer = Embedding(input_dim=num_most_freq_words_to_include,
                                output_dim=embedding_vector_length,
                                input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
    embout = embedding_layer(input_sequences)
    gruout = GRU(100, return_sequences=True)(embout)
    attout = attention_3d_block(gruout)
    outputs = Dense(1, activation='sigmoid')(attout)
    model = Model(inputs=input_sequences, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

## RNN model initization
gru_att_model = rnn_model()
gru_att_model.summary()

## training the rnn model
gru_att_model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=[x_val, y_val])
y_test_pred_gru_att = gru_att_model.predict(x_test)

## calculatin the accuracy and f1 score
print("The AUC socre for GRU attention model is : %.4f." %roc_auc_score(y_test, y_test_pred_gru_att.round()))
print("F1 score for GRU attention model is: %.4f." % f1_score(y_test, y_test_pred_gru_att.round()))