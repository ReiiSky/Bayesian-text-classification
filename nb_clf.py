import numpy
from core import var, array


import pandas

class NaiveBayes:
    def __init__(self, separate=" "):
        self.separate = separate #string
        self.prob = None # array of float
        self.unique_word = None #array of string
        self.feature_of_class = None #float
        self.probability_each_document_over_class = dict() # dictionary


    def _sentence_split(self, sentence):
        result = list()
        temp = numpy.char.split(sentence, sep=self.separate)
        print(temp.shape)
        for i in temp:
            result.extend(i)
        return array(result)

    def _unique_sentence(self, sentence):
        return numpy.unique(sentence)

    def _probability_each_class(self, class_):
        self.feature_of_class = len(class_)
        self.prob = dict()

        for i in class_:
            if i not in self.prob:
                outcome = len(class_[class_ == i])
                self.prob[i] = var(outcome)/var(self.feature_of_class)

    def _get_outcome_from_each_document(self, document, class_):
        self.unique_word = self._unique_sentence(self._sentence_split(document))
        keys = self.prob.keys()
        for key in keys:
            current_class_index: bool = class_ == key
            class_outcome = 0
            self.probability_each_document_over_class[key] = dict()
            for x in range(len(current_class_index)):
                if current_class_index[x]:
                    current_document_feature = self._sentence_split([document[x]])
                    for i in self.unique_word:
                        feature = current_document_feature == i
                        size = len(current_document_feature[feature])
                        class_outcome += size
                        self.probability_each_document_over_class[key][i] = size+1
            for j in self.unique_word:
                get = self.probability_each_document_over_class[key][j]
                self.probability_each_document_over_class[key][j] = self._compute_propability(get, class_outcome, len(self.unique_word))

    def _compute_propability(self, word_outcome, unique_over_class, length_of_unique):
        return var(word_outcome)/(var(unique_over_class)+var(length_of_unique))

    def fit(self, document, class_):
        self._probability_each_class(class_)
        self._get_outcome_from_each_document(document, class_)

    def free(self):
        del self.feature_of_class
        del self.unique_word

    def predict(self, x):
        x_split = self._sentence_split(x)
        key_max = ["", 0]
        for c_ in self.prob.keys():
            mult = var(self.prob[c_])
            for x_ in x_split:
                feature_document = array([*self.probability_each_document_over_class[c_].keys()])
                condition = feature_document == x_
                length = len(feature_document[condition])
                if length == 0:
                    continue
                prob = self.probability_each_document_over_class[c_][x_]
                mult *= var(prob)
            if mult > key_max[1]:
                key_max[0] = c_
                key_max[1] = mult
        return key_max[0]


def main():
    dictionary = array(["my name is reikaa", "what is your name ?"])
    class_ = array(["1", "2"])
    Classifier = NaiveBayes()
    Classifier.fit(dictionary, class_)
    print(Classifier.predict(["my name is entalizen"]))

if __name__ == "__main__":
    main()