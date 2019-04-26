# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019
import math
# import sys
# sys.path.insert(0, '/Users/hatal/OneDrive/Documents/UIUC/Junior_Year/CS440/mp3/CS440_MP3/part1')

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 1
        self.num_class = 14
        self.prior = [0 for i in range(14)]
        self.numWordsCount = [0 for i in range(14)]
        self.likelihood = [ {} for i in range(14) ]
        self.likelihoodbigram = [ {} for i in range(14) ]
        self.top_feature_words = []
        self.classNames = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        probModel = [ {} for i in range(14) ]
        probModel2 = [ {} for i in range(14) ]
        bagOWordsModel = [ {} for i in range(14) ]
        bagOTuplesModel = [ {} for i in range(14) ]

        idx = -1
        lpVal = 1
        numLabels = len(train_label)
        for i in train_label:
            self.prior[i - 1] += 1
        self.prior = [x/numLabels for x in self.prior]
        # print(self.prior)
        for words in train_set:
            idx += 1
            for i in range(len(words)):
                if words[i] not in bagOWordsModel[train_label[idx] - 1]:
                    for classType in range(14):
                        bagOWordsModel[classType][words[i]] = 0
                bagOWordsModel[train_label[idx] - 1][words[i]] += 1

                for j in range(i+1,len(words)):
                    tuple1 = [words[i], words[j]]
                    tuple1.sort()
                    tup = (tuple1[0],tuple1[1])
                    if tup not in bagOTuplesModel[train_label[idx] - 1]:
                        for classType in range(14):
                            bagOTuplesModel[classType][tup] = 0
                    bagOTuplesModel[train_label[idx] - 1][tup] += 1

        for i in range(14):
            for word in bagOWordsModel[i].keys():
                self.numWordsCount[i] += bagOWordsModel[i][word]
        sumCounts1 = sum(bagOWordsModel[0].values())
        sumCounts2 = sum(bagOTuplesModel[0].values())

        for i in range(14):
            for words in bagOWordsModel[i].keys():
                prob = (bagOWordsModel[i][words] + lpVal) / (sumCounts1*lpVal + self.prior[i]*numLabels)
                self.likelihood[i][words] = prob
            for tup in bagOTuplesModel[i].keys():
                prob = (bagOTuplesModel[i][tup] + lpVal) / (sumCounts2*lpVal + self.prior[i]*numLabels)
                self.likelihoodbigram[i][tup] = prob
        # for i in range(0, self.num_class):
        #     curr_dict = self.likelihood[i]
        #     curr_dict = sorted(curr_dict, key=curr_dict.get, reverse=True)[:20]
        #     self.top_feature_words.append(curr_dict)
        # print(self.top_feature_words)
        # self.likelihood = probModel
        # print(self.likelihood[0])
        # TODO: Write your code here
        pass

    def predict(self, x_set, dev_label,lambda_mix=1):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text

        """
        idx = -1
        accuracy = 0.0
        result = []
        for document in x_set:
            idx += 1
            probs1 = [0 for x in range(14)]
            probs2 = [0 for x in range(14)]
            probs = [0 for x in range(14)]

            for classType in range(14):
                for i in range(len(document)):
                    if document[i] in self.likelihood[classType]:
                        probs1[classType] += math.log(self.likelihood[classType][document[i]])
                    elif document[i] not in self.likelihood[classType]:
                        probs1[classType] += math.log(1/(len(self.likelihood[classType].keys()) + self.numWordsCount[classType]))

                    for j in range(i+1,len(document)):
                        tuple1 = [document[i], document[j]]
                        tuple1.sort()
                        tup = (tuple1[0],tuple1[1])
                        if tup in self.likelihoodbigram[classType]:
                            probs2[classType] += math.log(self.likelihoodbigram[classType][tup])
                        # elif tup not in self.likelihoodbigram[classType]:
                        #     probs2[classType] += math.log(1/self.prior[classType])

                probs1[classType] += math.log(self.prior[classType])
                probs2[classType] += math.log(self.prior[classType])
                probs[classType] = probs1[classType] * (1 - self.lambda_mixture) + probs2[classType] * self.lambda_mixture
            result.append(probs.index(max(probs)) + 1)
            if dev_label[idx] == result[idx]:
                accuracy += 1
        # print (accuracy)
        accuracy /= len(dev_label)
        print("Accuracy: ", accuracy)
        # TODO: Write your code here
        pass

        return accuracy,result
