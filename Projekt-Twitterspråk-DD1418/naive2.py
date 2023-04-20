import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_curve, auc, plot_roc_curve
from sklearn.calibration import calibration_curve

# Load the training and testing datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def optimize_params(X_train, y_train):
    # Define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])

    # Define the parameter grid to search over
    parameters = {
        'vect__ngram_range': [(1,1), (1,2), (1,3)],
        'clf__alpha': [0.1, 1, 10]
    }

    # Perform grid search cross-validation to find the best parameters
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and their corresponding mean cross-validation score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    return grid_search

train_data = pd.read_csv('AirlineTweets_training.csv', encoding="UTF-8")
test_data = pd.read_csv('AirlineTweets_test.csv', encoding="UTF-8", skiprows=1468)

# Prompt user for choice of ngram range
n = int(input("Enter the number of words to consider in the n-gram model (1 for unigrams, 2 for bigrams, 3 for trigrams): "))

# Create a CountVectorizer object to convert text into numerical features
if n == 1:
    vectorizer = CountVectorizer(ngram_range=(1,1))
elif n == 2:
    vectorizer = CountVectorizer(ngram_range=(2,2))
elif n == 3:
    vectorizer = CountVectorizer(ngram_range=(3,3))
else:
    print("Invalid input. Defaulting to unigrams.")
    vectorizer = CountVectorizer(ngram_range=(1,1))

# Fit the vectorizer on the training data
vectorizer.fit(train_data.iloc[:,1])

# Transform the text into numerical features for training and testing data
X_train = vectorizer.transform(train_data.iloc[:,1])
y_train = train_data.iloc[:,0]
X_test = vectorizer.transform(test_data.iloc[:,1])
y_test = test_data.iloc[:,0]

# encode the labels as numeric values
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Train a Multinomial Naive Bayes classifier on the training data
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the performance of the classifier using F1 score for each class
report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
print(report)

# Compute ROC curve and AUC for Neutral vs Negative sentiment
fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Neutral vs Negative sentiment')
plt.legend(loc="lower right")
plt.show()

# Compute ROC curve and AUC for Positive vs Negative sentiment
fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=2)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Positive vs Negative sentiment')
plt.legend(loc="lower right")
plt.show()

# Plot the calibration curve
prob_pos, prob_true = calibration_curve(y_test == 'negative', classifier.predict_proba(X_test)[:, 0], n_bins=10)

# Plot the calibration curve
plt.plot(prob_true, prob_pos, marker='o')
plt.title('Calibration Curve')
plt.xlabel('True Probability')
plt.ylabel('Fraction of Positives')
plt.show()

