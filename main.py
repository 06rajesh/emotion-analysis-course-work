import dict_emo
import numpy as np
from py_isear.isear_loader import IsearLoader
from py_isear.enums import EMOT
from ssec_dataset.loader import Loader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score

import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


dict_labels = ["joy", "anger", "surprise", "sadness", "fear"]


class OneHotEncoder:

    def __init__(self, classes: list):
        self.classes = classes

    def encode(self, outputs):
        encoded_outputs = []
        for row in outputs:
            encoded_row = [0] * len(self.classes)
            for idx, label in enumerate(self.classes):
                if label in row:
                    encoded_row[idx] = 1
            encoded_outputs.append(encoded_row)
        return encoded_outputs


def compute_accuracy(y_pred, y_true):
    enc = OneHotEncoder(dict_labels)
    pred_enc = enc.encode(y_pred)
    true_enc = enc.encode(y_true)
    pred_enc = np.array(pred_enc)
    true_enc = np.array(true_enc)
    for idx, label in enumerate(dict_labels):
        label_pred = pred_enc[:, idx]
        label_true = true_enc[:, idx]
        f1 = f1_score(label_pred, label_true, average='macro')
        print("%s : %2f" % (label, f1))


def run_isear_dict_emo():
    data = ['TEMPER', 'TROPHO']
    target = ['EMOT']
    loader = IsearLoader(data, target)
    dataset = loader.load_isear('isear.csv')

    text_data_set = dataset.get_freetext_content()
    target_set = dataset.get_target_emot_labels()
    n_data = len(text_data_set)

    n_train_data = n_data - int(n_data * 0.1)
    x_train = text_data_set[:n_train_data]
    y_train = target_set[:n_train_data]
    x_test = text_data_set[n_train_data:]
    y_test = target_set[n_train_data:]

    output = dict_emo.predict(x_test)
    compute_accuracy(output, y_test)


def run_isear_sgd_classifier():
    data = ['TEMPER', 'TROPHO']
    target = ['EMOT']
    loader = IsearLoader(data, target)
    dataset = loader.load_isear('isear.csv')

    text_data_set = dataset.get_freetext_content()
    target_set = dataset.get_target()
    target_chain = itertools.chain(*target_set)
    target_data = list(target_chain)

    x_train, x_test, y_train, y_test = train_test_split(text_data_set, target_data, test_size=0.1, random_state=42)

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier()),
                         ])

    fitting = text_clf.fit(x_train, y_train)
    y_pred = text_clf.predict(x_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    conf = confusion_matrix(y_test, y_pred)

    print("Macro: {}".format(f1_macro))
    print("Micro: {}".format(f1_micro))
    print(conf)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)


def run_ssec_dict_emo():
    loader = Loader()
    loader.load()

    x_test, y_test = loader.get_test_data()
    output = dict_emo.predict(x_test)
    compute_accuracy(output, y_test)


def run_ssec_sgd_classifier():
    loader = Loader()
    loader.load()

    x_train, y_train = loader.get_train_data()
    classes = loader.get_classes()
    x_test, y_test = loader.get_test_data()

    enc = OneHotEncoder(classes)
    y_train_encoded = enc.encode(y_train)
    y_train_encoded = np.array(y_train_encoded)

    y_test_encoded = enc.encode(y_test)
    y_test_encoded = np.array(y_test_encoded)

    print("\t\tPr\tRe\tMacro\tMicro\tAcc")
    for i in range(len(classes)):
        y = y_train_encoded[:, i]
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier()),
                             ])

        fitting = text_clf.fit(x_train, y)
        y_pred = text_clf.predict(x_test)
        y_true = y_test_encoded[:, i]
        f1 = f1_score(y_pred, y_true, average='macro')
        f1_micro = f1_score(y_pred, y_true, average='micro')
        pr = precision_score(y_true, y_pred)
        re = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print("{}\t\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(classes[i], pr, re, f1, f1_micro, acc))


if __name__ == '__main__':
    run_isear_sgd_classifier()
    # run_ssec_sgd_classifier()




