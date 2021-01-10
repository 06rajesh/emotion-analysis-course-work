import numpy as np
import itertools
import os
import time
import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, InputLayer
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


from py_isear.isear_loader import IsearLoader
from ssec_dataset.loader import Loader
from embeddings import Glove
from main import OneHotEncoder

EMBEDDING_DIM = 200


def process_inputs(x):
    word_sequences = []
    max_sentence_len = 0

    for i in x:
        i = str(i)
        i = i.replace('\'', '')
        newlist = [x for x in text_to_word_sequence(i, lower=True)]
        if len(newlist) > max_sentence_len:
            max_sentence_len = len(newlist)
        word_sequences.append(newlist)
        pass

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_sequences)
    word_indices = tokenizer.texts_to_sequences(word_sequences)
    word_index = tokenizer.word_index

    # padding word_indices
    x_data = pad_sequences(word_indices, maxlen=max_sentence_len)
    print("After padding data")
    print(x_data.shape)
    return x_data, word_index, max_sentence_len


def process_labels(y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print("Label Encoding Classes as ")
    print(le_name_mapping)

    y_data = to_categorical(integer_encoded)
    print("One Hot Encoded class shape ")
    print(y_data.shape)
    return y_data


def create_model(input_layer, n_outputs, draw_model=True):
    model = Sequential()
    model.add(input_layer)
    model.add(Conv1D(30, 1, activation="relu"))
    model.add(MaxPooling1D(4))
    model.add(LSTM(100, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    if draw_model:
        plot_model(
            model, to_file='model.png', show_shapes=True,
            show_layer_names=False, rankdir='TB', expand_nested=True, dpi=96
        )

    return model


def run_lstm_model(x, y, process_y=True, class_label=None):
    x_data, word_index, max_sentence_len = process_inputs(x)

    # creating embedding matrix
    embed = Glove()
    embed.load()

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embed.get_embedding_val(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Embedding Matrix Generated : ", embedding_matrix.shape)

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=max_sentence_len, trainable=False)

    if process_y:
        y_data = process_labels(y)
        print("Finished Preprocessing data ...")
    else:
        y_data = y

    print("spliting data into training, testing set")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

    batch_size = 64
    num_epochs = 50
    x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
    x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

    print(y_data.shape)
    output_layer = y_data.shape[1]
    model = create_model(embedding_layer, output_layer)

    checkpath = "checkpoints/isear/model_weights.ckpt"
    checkpoint = ModelCheckpoint(checkpath, save_weights_only=True, verbose=1)
    callbacks_list = [checkpoint]

    # history = model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size,
    #                    epochs=num_epochs, callbacks=callbacks_list)

    checkdir = os.path.dirname(checkpath)
    latest = tf.train.latest_checkpoint(checkdir)
    model.load_weights(latest)
    scores = model.evaluate(x_test, y_test, verbose=0)

    y_pred = model.predict(x_test)
    compute_accuracy(y_test, y_pred)
    # compute_ssec_accuracy(y_test, y_pred)

    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #
    # plt.legend()
    # plt.show()


def compute_accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    print("Macro: {}".format(f1_macro))
    print("Micro: {}".format(f1_micro))
    con = confusion_matrix(y_true, y_pred)
    print(con)

    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred)
    print(report)


def compute_ssec_accuracy(y_true, y_pred):
    y_labeled = np.where(y_pred > 0.5, 1, 0)

    print("\t\tPr\tRe\tMacro\tMicro\tAcc")
    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_labeled[:, i]
        f1 = f1_score(y_t, y_p, average='macro')
        f1_micro = f1_score(y_t, y_p, average='micro')
        pr = precision_score(y_t, y_p)
        re = recall_score(y_t, y_p)
        acc = accuracy_score(y_t, y_p)
        print("{}\t\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(classes[i], pr, re, f1, f1_micro, acc))


if __name__ == '__main__':
    data = ['TEMPER', 'TROPHO']
    target = ['EMOT']
    loader = IsearLoader(data, target)
    dataset = loader.load_isear('isear.csv')

    text_data_set = dataset.get_freetext_content()
    target_set = dataset.get_target()
    target_chain = itertools.chain(*target_set)
    target_data = list(target_chain)

    run_lstm_model(text_data_set, target_data)

    loader = Loader()
    loader.load()

    x_1, y_1 = loader.get_train_data()
    classes = loader.get_classes()
    x_2, y_2 = loader.get_test_data()
    text_data_set = x_1 + x_2
    target_data = y_1 + y_2
    # enc = OneHotEncoder(classes)
    # y_train_encoded = enc.encode(target_data)
    # y_train_encoded = np.array(y_train_encoded)
    #
    # run_lstm_model(text_data_set, y_train_encoded, process_y=False)
