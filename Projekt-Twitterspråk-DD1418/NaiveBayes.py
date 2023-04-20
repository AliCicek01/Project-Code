import numpy as np
import re
import pandas as pd
import math
from tabulate import tabulate
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class NaiveBayes:

    def __init__(self, dfTraining,dfTest,dict_index,demo,process=True):
        self.dict_index=dict_index
        self.positiveDictSize = 0
        self.negativeDictSize = 0
        self.neutralDictSize = 0
        self.positive = defaultdict(int)
        self.negative = defaultdict(int)
        self.neutral = defaultdict(int)
        self.total = defaultdict(int)
        self.neutral_tweet_count= 0
        self.negative_tweet_count = 0
        self.positive_tweet_count = 0
        self.total_tweet_count = 0
        self.dicts = [self.neutral, self.positive, self.negative]
        self.predicted_labels = []
        self.correct_labels = []
        self.process_data(dfTraining)
        self.compute_probs(dfTest,demo)
        self.evaluation()

    def process_data(self, dataframe):
        for index, row in dataframe.iterrows():
            tweet = row['tweet']
            label_index = self.dict_index[row['label']]
            self.total_tweet_count += 1

            # Update tweet sentiment counts
            if label_index == 0:
                self.neutral_tweet_count += 1
            elif label_index == 1:
                self.positive_tweet_count += 1
            elif label_index == 2:
                self.negative_tweet_count += 1

            tweet = tweet.lower()
            tweet = re.sub('[^A-Za-z ]+', '', tweet)
            word_list = tweet.split()

            for word in word_list:
                # Update word counts for each sentiment category
                self.dicts[label_index][word] += 1
                self.total[word] += 1

        # Compute sizes of each sentiment category dictionary and the total unique words count
        self.positive_dict_size = sum(self.positive.values())
        self.negative_dict_size = sum(self.negative.values())
        self.neutral_dict_size = sum(self.neutral.values())
        self.total_unique_words_count = len(self.total)

    def compute_probs(self, dataframe, demo=False):
        for index, row in dataframe.iterrows():
            tweet = row['tweet']
            label_index = self.dict_index[row['label']]
            self.correct_labels.append(label_index)

            tweet = tweet.lower()
            tweet = re.sub('[^A-Za-z ]+', '', tweet)
            word_list = tweet.split()
            probs = np.zeros(3)

            # Compute prior probabilities
            probs[0] = math.log(self.neutral_tweet_count / self.total_tweet_count)
            probs[1] = math.log(self.positive_tweet_count / self.total_tweet_count)
            probs[2] = math.log(self.negative_tweet_count / self.total_tweet_count)

            # Compute conditional probabilities
            for word in word_list:
                probs[0] += math.log(
                    (self.neutral[word] + 1) / (self.neutral_dict_size + self.total_unique_words_count))
                probs[1] += math.log(
                    (self.positive[word] + 1) / (self.positive_dict_size + self.total_unique_words_count))
                probs[2] += math.log(
                    (self.negative[word] + 1) / (self.negative_dict_size + self.total_unique_words_count))

            self.predicted_labels.append(probs.argmax())

        # If demo flag is set, call demo method to display predicted and correct labels
        if demo == True:
            self.demo(self.predicted_labels, self.correct_labels, dataframe)

    def evaluation(self):
        conf_mat = confusion_matrix(self.correct_labels, self.predicted_labels)
        headers = ['Neutral', 'Positive', 'Negative']
        new_conf_mat = []
        for i in range(len(headers)):
            temp_list = [headers[i], conf_mat[i, 0], conf_mat[i, 1], conf_mat[i, 2]]
            new_conf_mat.append(temp_list)

        table = tabulate(new_conf_mat, headers, tablefmt='fancy_grid')
        print("Confusion Matrix:")
        print(table)

        recall_list = recall_score(self.correct_labels, self.predicted_labels, average=None)
        precision_list = precision_score(self.correct_labels, self.predicted_labels, average=None)
        tkr = 0
        for i in range(len(self.correct_labels)):
            if int(self.correct_labels[i]) == self.predicted_labels[i]:
                tkr += 1

        acc = (tkr / len(self.correct_labels)) * 100
        print(f"Accuracy is {acc}%")
        print(f"Precision Neutral: {precision_list[0]}, Positive: {precision_list[1]}, Negative: {precision_list[2]}")
        print(f"Recall    Neutral: {recall_list[0]}, Positive: {recall_list[1]}, Negative: {recall_list[2]}")
        print(f"F-score    Neutral: {2 * recall_list[0] * precision_list[0] / (recall_list[0] + precision_list[0])},"
              f" Positive: {2 * recall_list[1] * precision_list[1] / (recall_list[1] + precision_list[1])}, "
              f"Negative: {2 * recall_list[2] * precision_list[2] / (recall_list[2] + precision_list[2])}")



        acc = (tkr/(len(self.correct_labels)-1))*100
        print(f"Accuracy is {acc}%")
        print(f"Precision Neutral: {precision_list[0]}, Positive: {precision_list[1]}, Negative: {precision_list[2]}")
        print(f"Recall    Neutral: {recall_list[0]}, Positive: {recall_list[1]}, Negative: {recall_list[2]}")
        print(f"F-score    Neutral: {2*recall_list[0]*precision_list[0]/(recall_list[0]+precision_list[0])},"
              f" Positive: {2*recall_list[1]*precision_list[1]/(recall_list[1]+precision_list[1])}, "
              f"Negative: {2*recall_list[2]*precision_list[2]/(recall_list[2]+precision_list[2])}")



    def demo(self, model_labels, cor_labels, df):
        for index, line in df.iterrows():
            if 10 < index < 20:
                print(f"Tweet: {line['tweet']}, Actuall: {cor_labels[index]}, Predicted: {model_labels[index]}")

def main():
    #If you want to run AirlineTweets

    dict_index = {'neutral': 0, 'positive': 1, 'negative': 2}
    dataset_columns = ['label', 'tweet']
    dfTraining = pd.read_csv("AirlineTweets_training.csv", encoding="UTF-8", names=dataset_columns)
    dfTest = pd.read_csv("AirlineTweets_test.csv", encoding="UTF-8", names=dataset_columns)
    NaiveBayes(dfTraining, dfTest, dict_index, demo=True)


    #If you want to run RandomTweets
    """
    dict_index = {0: 0, 1: 1, -1: 2}
    dataset_columns = ['id','tweet', 'label']
    dfTraining = pd.read_csv("RandomTweets_TrainingData.csv", encoding="UTF-8", names=dataset_columns)
    dfTest = pd.read_csv("RandomTweets_TestingData.csv", encoding="UTF-8", names=dataset_columns)
    NaiveBayes(dfTraining,dfTest,dict_index,demo=False)
    """



if __name__ == '__main__':
    main()