import csv
import os.path as path


class Loader:

    def __init__(self):
        self.dirname = path.join(path.dirname(path.abspath(__file__)), "corpora")
        self.trainfile = path.join(self.dirname, "train-combined-0.0.csv")
        self.testfile = path.join(self.dirname, "test-combined-0.0.csv")
        self.train_x, self.train_y  = list(), list()
        self.test_x, self.test_y = list(), list()
        self.classes = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    @staticmethod
    def load_csv(filename: str):
        x, y = list(), list()
        line_count = 0
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')

            for row in csv_reader:
                n_col = len(row)
                labels = list()
                if n_col == 9:
                    for i in range(n_col - 1):
                        if row[i] != "---":
                            label = row[i].lower()
                            labels.append(label)
                    x.append(row[n_col - 1])
                    y.append(labels)
                    line_count += 1
        print("Total %2d Lines have been loaded" % line_count)
        return x, y

    def load(self):
        self.train_x, self.train_y = self.load_csv(self.trainfile)
        self.test_x, self.test_y = self.load_csv(self.testfile)

    def get_classes(self):
        return self.classes

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y


