import numpy as np
import itertools
from py_isear.isear_loader import IsearLoader

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


if __name__ == '__main__':
    data = ['TEMPER', 'TROPHO']
    target = ['EMOT']
    loader = IsearLoader(data, target)
    dataset = loader.load_isear('isear.csv')

    text_data_set = dataset.get_freetext_content()
    target_set = dataset.get_target_emot_labels()
    target_chain = itertools.chain(*target_set)
    target_data = list(target_chain)
    n_data = len(text_data_set)

    n_train_data = n_data - int(n_data * 0.1)
    x_train = text_data_set[:n_train_data]
    y_train = target_data[:n_train_data]
    x_test = text_data_set[n_train_data:]
    y_test = target_data[n_train_data:]

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier()),
                         ])

    fitting = text_clf.fit(x_train, y_train)
    y_pred = text_clf.predict(x_test)
    f1 = f1_score(y_pred, y_test, average='micro')
    print(f1)


