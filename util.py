
from bs4 import BeautifulSoup
import re, numpy as np
import pandas as pd

from _text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

stopwrds = open('stopword.txt').read().split('\n')

def html2text(review):
    """Return extracted text string from provided HTML string."""
    review_text = BeautifulSoup(review, "lxml").get_text()
    if len(review_text) == 0:
        review_text = review
    review_text = re.sub(r"\<.*\>", "", review_text)
    try:
        review_text = review_text.encode('ascii', 'ignore').decode('ascii')#ignore \xc3 etc.
    except UnicodeDecodeError:
        review_text = review_text.decode("ascii", "ignore")
    return review_text


def letters_only(text):
    """Return input string with only letters (no punctuation, no numbers)."""
    # It is probably worth experimenting with milder prepreocessing (eg just removing punctuation)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(r"[ ]{2,}", ' ', text)
    text2 = []
    for i in text.split(' '):
        if i not in stopwrds:
            text2.append(i)
    return ' '.join(text2)

def review_preprocess(review):
    """Preprocessing used before fitting/transforming RNN tokenizer - Html->text, remove punctuation/#s, lowercase."""
    return letters_only(str(html2text(review)).lower())

def get_train_data(df, reviews_to_features_fn=None):
    """Extracts features (using reviews_to_features_fn), splits into train/test data, and returns
    x_train, y_train, x_test, y_test.  If no feature extraction function is provided, x_train/x_test will
    simply consist of a Series of all the reviews.
    """
#     df = pd.read_csv('labeledTrainData.tsv', header=0, quotechar='"', sep='\t')
    SEED = 1000
    # Shuffle data frame rows
    np.random.seed(SEED)
    df = df.iloc[np.random.permutation(len(df))]

    if reviews_to_features_fn:
        feature_rows = df["review"].map(reviews_to_features_fn)
        if type(feature_rows[0]) == np.ndarray:
            num_instances = len(feature_rows)
            num_features = len(feature_rows[0])
            x = np.concatenate(feature_rows.values).reshape((num_instances, num_features))
        else:
            x = feature_rows
    else:
        x = df["review"]

    y = df["sentiment"]

    # Split 80/20
    test_start_index = int(df.shape[0] * .8)
    x_train = x[0:test_start_index]
    y_train = y[0:test_start_index]
    x_val = x[test_start_index:]
    y_val = y[test_start_index:]

    return x_train, y_train, x_val, y_val

def preprocess(train, test, min_word_count=0, num_most_freq_words_to_include = None):
    x_train, y_train, x_val, y_val = get_train_data(train, review_preprocess)
    x_test = test["review"].map(review_preprocess)
    test["sentiment"] = test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
    y_test = test["sentiment"]
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    train_review_list = x_train.tolist()
    val_review_list = x_val.tolist()
    test_review_list = x_test.tolist()
    all_review_list = x_train.tolist() + x_val.tolist()

    np.random.seed(1000)

    tokenizer = Tokenizer(min_word_count=min_word_count)
    # print all_review_list[0:2]
    tokenizer.fit_on_texts(all_review_list)

    train_reviews_tokenized = tokenizer.texts_to_sequences(train_review_list)
    x_train = train_reviews_tokenized
    # x_train = pad_sequences(train_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
    val_review_tokenized = tokenizer.texts_to_sequences(val_review_list)
    x_val = val_review_tokenized
    # x_val = pad_sequences(val_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
    test_review_tokenized = tokenizer.texts_to_sequences(test_review_list)
    x_test = test_review_tokenized
    # x_test = pad_sequences(test_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

    word2idx = tokenizer.word_index
    word2idx.update({'UNKOWN':0})
    idx2word = {v:k for k,v in word2idx.items()}

    return x_train, y_train, x_val, y_val, x_test, y_test, word2idx, idx2word

def init_weight(Mi, Mo=0):
    if Mo == 0:
        return np.random.randn(Mi)/np.sqrt(Mi)
    return np.random.randn(Mi, Mo)/np.sqrt(Mi + Mo)


if __name__ == "__main__":
    train = pd.read_csv("/media/zero/41FF48D81730BD9B/kaggle/word2vec-nlp/input/labeledTrainData.tsv", header=0,delimiter='\t')
    test = pd.read_csv("/media/zero/41FF48D81730BD9B/kaggle/word2vec-nlp/input/testData.tsv", header=0, delimiter='\t')

    x_train, y_train, x_val, y_val, x_test, y_test, word2idx, idx2word = preprocess(train, test)