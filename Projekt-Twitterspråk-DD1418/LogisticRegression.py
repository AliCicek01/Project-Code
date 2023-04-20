import pandas as pd
import numpy as np
from collections import defaultdict
import math
import re
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.special import softmax as softmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
"""
Sentiment Analysis for tweets. Three different classes: Neutral, Positive, Negative
See README for more information on how to run

"""

class SentimentAnalysis:

    learning_rate = 0.01
    convergence_margin = 0.0001
    max_iterations = 1000

    def __init__(self, df, conv_dict, dataset_encoding, dataset_columns, should_train=True):
        self.dataset_columns = dataset_columns
        self.dataset_encoding = dataset_encoding
        self.num_datapoints = len(df)
        self.num_features = 12
        self.num_classes = 3
        self.conv_dict = conv_dict

        # Convert various dictionaries to lookup tables
        self.neg_words = self.convert_file_to_dict('negative-words.txt')
        self.pos_words = self.convert_file_to_dict('positive-words.txt')
        self.pos_sup = self.convert_file_to_dict('positive-superlatives.txt')
        self.neg_sup = self.convert_file_to_dict('negative-superlatives.txt')
        self.pos_adv = self.convert_file_to_dict('positive-adverbs.txt')
        self.neg_adv = self.convert_file_to_dict('negative-adverbs.txt')
        self.pos_adj = self.convert_file_to_dict('positive-adjectives.txt')
        self.neg_adj = self.convert_file_to_dict('negative-adjectives.txt')
        #self.emojis = self.convert_file_to_dict('emojis.txt')

        # Initialize theta and gradient with random values
        self.theta = np.random.uniform(-1, 1, [self.num_classes, self.num_features])
        self.gradient = np.zeros([self.num_classes, self.num_features])

        if should_train: # if should_train -> train model
            self.x = self.process_data(df['tweet'])
            self.y = self.convert_labels(df['label'])
            self.stochastic_fit()

    def stochastic_gradient(self, datapoint):
        # Compute the gradient for a single datapoint using stochastic gradient descent
        theta_x = np.matmul(self.theta, self.x[:, datapoint])
        softmax_values = softmax(theta_x, axis=0)
        softmax_values = np.transpose(softmax_values)
        softmax_y = softmax_values - self.y[datapoint]

        for j in range(len(softmax_values)):
            self.gradient[j] = self.x[:, datapoint] * (softmax_y[j])

    def stochastic_fit(self):
        counter = 0
        i = 0
        self.init_plot(2)
        while counter < self.max_iterations:
            self.stochastic_gradient(np.random.randint(0, self.num_datapoints))
            self.theta = self.theta - self.learning_rate * self.gradient

            if i % 11 == 0:
                self.update_plot(self.loss(self.x, self.y))

            i += 1
            counter += 1

    def compute_gradient(self):
        # Compute the gradient using all datapoints
        theta_x = np.matmul(self.theta, self.x)
        softmax_values = softmax(theta_x, axis=0)
        softmax_values = np.transpose(softmax_values)
        softmax_y = softmax_values - self.y

        for k in range(self.num_features):
            grad = np.asarray([self.x[k, :] * softmax_y[:, c] for c in range(self.num_classes)])
            self.gradient[:, k] = np.sum(grad, axis=1)
        self.gradient = self.gradient / self.num_datapoints

    def loss(self, inputs, targets):
        total_loss = 0

        theta_x = np.matmul(self.theta, inputs)  # Calculate all dot products between inputs and theta
        sig = softmax(theta_x, axis=0)  # Apply softmax activation function

        for i in range(self.num_datapoints):
            j = np.argmax(targets[i, :])
            total_loss -= math.log(sig[j, i])  # Calculate loss using cross-entropy

        average_loss = total_loss / self.num_datapoints
        return average_loss

    def fit(self):
        epsilon = 1
        iteration = 0
        self.init_plot(2)

        while epsilon > self.convergence_margin:
            self.compute_gradient()
            self.theta = self.theta - self.learning_rate * self.gradient
            epsilon = np.sum(self.gradient ** 2)  # Calculate the Euclidean length of the gradient

            if iteration % 11 == 0:
                self.update_plot(self.loss(self.x, self.y))  # Update plot with the current loss

            iteration += 1

    def convert_file_to_dict(self, filename):
        # Converts file to Dict
        file = open(filename, 'r')
        words = file.readlines()
        dct = defaultdict()

        for word in words:
            word = word.strip()
            dct[word] = 1

        return dct

    def process_data(self, dataset): # Fills the x matrix with values
        i = 0
        x_mat = np.zeros([self.num_features, self.num_datapoints])

        for line in dataset:
            tweet = line

            if type(tweet) != str:
                tweet = str(tweet)

            cleaned = tweet.lower()
            cleaned = re.sub('[^A-Za-z ]+', '', cleaned)
            word_list = cleaned.split()
            word_list = word_list[1:]

            x_mat[0, i] = 1
            x_mat[1, i] = self.corpus_lookup(word_list, self.neg_words) # Negative word
            x_mat[2, i] = self.corpus_lookup(word_list, self.pos_words) # Positive word
            x_mat[3, i] = self.pos_sum(word_list) # Net sentoment positive
            x_mat[4, i] = self.neut_sum(word_list) # Net sentiment neutral
            x_mat[5, i] = self.tweet_length(word_list) # Length of tweet
            x_mat[6, i] = self.corpus_lookup(word_list, self.pos_sup) # Positive superlatives
            x_mat[7, i] = self.corpus_lookup(word_list, self.neg_sup) # Negative superlatives
            x_mat[8, i] = self.corpus_lookup(word_list, self.pos_adv) # Positive adverbs
            x_mat[9, i] = self.corpus_lookup(word_list, self.neg_adv) # Negative adverbs
            x_mat[10, i] = self.corpus_lookup(word_list, self.pos_adj) # Positive adjectives
            x_mat[11, i] = self.corpus_lookup(word_list, self.neg_adj) # Negative adjectives
            #x_mat[12, i] = self.emoji_feature(word_list, self.emojis) # Emojis



            i += 1

        return x_mat

    # FEATURES BELOW

    def pos_sum(self, word_list):
        sum = 0

        for word in word_list:
            try:
                if self.neg_words[word] == 1:
                    sum -= 1
                elif self.pos_words[word] == 1:
                    sum +=1
            except KeyError:
                pass
        if sum > 0:
            return 1
        else:
            return 0

    def neut_sum(self, word_list):
        sum = 0

        for word in word_list:
            try:
                if self.neg_words[word] == 1:
                    sum -= 1
                elif self.pos_words[word] == 1:
                    sum +=1
            except KeyError:
                pass
        if sum == 0:
            return 1
        else:
            return 0

    def tweet_length(self, word_list):

        if len(word_list) > 25:
            return 1
        else:
            return 0

    def corpus_lookup(self, word_list, dict):
        # Used for all lookups in corpuses

        for word in word_list:
            try:
                if dict[word] == 1:
                    return 1
            except KeyError:
                pass
        return 0

    def emoji_feature(self, word_list, dict):
        for word in word_list:
            try:
                if dict[word] == 1:
                    print(word)
                    return 1
            except KeyError:
                pass
        return 0

    # END OF FEATURES

    def test(self, test_file, weights=None, demo=False): # Optional argument for theta to save time and not compute gradient
        df = pd.read_csv(test_file, encoding=self.dataset_encoding, names=self.dataset_columns)
        headers = ['deadweight', 'neg word', 'pos word', 'pos sum', 'neut sum', 'length', 'pos sup',
                   'neg sup','pos adv', 'neg adv', 'pos adj', 'neg adj', "emojis"]

        table = tabulate(self.theta, headers, tablefmt='fancy_grid')
        print("Weight Matrix")
        print(table)

        # Provides correct labels
        cor_labels = np.zeros(len(df))
        j = 0
        for cor in df['label']:
            cor_labels[j] = self.conv_dict[cor]
            j += 1

        model_labels = []
        dataset = df['tweet']
        x_mat = self.process_data(dataset)

        if weights is not None:  # Fetches weights if provided
            self.theta = weights

        theta_x = np.matmul(self.theta, x_mat)  # All dot products between x and theta
        probs = softmax(theta_x, axis=0)

        for m in range(len(dataset)):
            predicted = np.argmax(probs[:,m])
            model_labels.append(predicted)


        tkr=0 # Calculates accuracy
        for i in range(0, len(dataset)):
            if int(cor_labels[i]) == model_labels[i]:
                tkr+=1

        acc = (tkr/(len(dataset)-1))*100

        # Confusion matrix
        print(f"Accuracy is {acc}%")
        print("Confusion Matrix:")
        conf_mat=confusion_matrix(cor_labels,model_labels)
        headers = [ 'Neutral', 'Positive', 'Negative']

        new_conf_mat=[]
        for i in range(len(headers)):
            temp_list=[headers[i], conf_mat[i,0], conf_mat[i,1], conf_mat[i,2]]

            new_conf_mat.append(temp_list)

        table = tabulate(new_conf_mat, headers, tablefmt='fancy_grid')
        print(table)
        recall_list = recall_score(cor_labels,model_labels,average=None)
        precision_list = precision_score(cor_labels,model_labels,average=None)
        print(f"Precision Neutral: {precision_list[0]}, Positive: {precision_list[1]}, Negative: {precision_list[2]}")
        print(f"Recall    Neutral: {recall_list[0]}, Positive: {recall_list[1]}, Negative: {recall_list[2]}")
        print(f"F-score    Neutral: {2*recall_list[0]*precision_list[0]/(recall_list[0]+precision_list[0])},"
              f" Positive: {2*recall_list[1]*precision_list[1]/(recall_list[1]+precision_list[1])}, "
              f"Negative: {2*recall_list[2]*precision_list[2]/(recall_list[2]+precision_list[2])}")

        if demo: # Prints a small demo
            self.demo(x_mat,model_labels,cor_labels,df)

    def demo(self, x_mat, model_labels, cor_labels, df):
        for index, line in df.iterrows():
            if 10 < index < 20:
                print(f"Tweet: {line['tweet']}, Actuall: {cor_labels[index]}, Predicted: {model_labels[index]}")


    def convert_labels(self, data): # Creates a y-matrix for the training set
        # One column for every possible class
        # All will be set to zero except for the true class
        y = np.zeros([len(data), self.num_classes])
        i = 0

        for line in data:
            j = self.conv_dict[line]
            y[i, j] = 1
            i += 1

        return y

    def init_plot(self, numaxes):
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(numaxes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    def update_plot(self, *args):
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)



def main():
    """
    #If you want to eneter weights manualy, just add a weight arguemnt to the tes method call
    weights = np.array([[-0.19, -0.46 -0.21, -0.12, -0.40, 0.42, -0.15],
                        [-0.95, -0.49, 1.09, 0.25, -0.004, 0.18, 0.55, 0.78, 0.53, 0.20, -0.70, -0.43, -0.09],
                        [0.30, 1.24, -0.36, 0.74, -0.09, 0.35, -0.30, -0.28, 0.07, 0.89, 0.07, 0.44, -0.61]])
    """

    #If you want to run RandomTweets
    """
    dataset_columns =['id','tweet','label']
    dataset_encoding = "UTF-8"
    df = pd.read_csv('RandomTweets_TrainingData.csv', encoding=dataset_encoding, names=dataset_columns)
    conv_dict = {0:0, 1:1, -1:2}
    sen = SentimentAnalysis(df, conv_dict, dataset_encoding, dataset_columns)
    sen.test("RandomTweets_TestingData.csv",demo=True)
    """

    #If you want to run AirlineTweets

    dataset_columns =['label', 'tweet']
    dataset_encoding = "UTF-8"
    df = pd.read_csv('AirlineTweets_training.csv', encoding=dataset_encoding, names=dataset_columns)
    conv_dict = {'neutral': 0, 'positive': 1, 'negative': 2}
    sen = SentimentAnalysis(df, conv_dict, dataset_encoding, dataset_columns)
    sen.test("AirlineTweets_test.csv")




if __name__ == '__main__':
    main()