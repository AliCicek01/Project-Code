
LogisticRegression:

To run, simply run LogisticRegression.py without any arguments.

Can run two different datasets, runs airline data by default.
Uncomment the other data and comment out the processing of the airline data in main if the other dataset should be run.

Trains by default, goes pretty quickly even for large datasets (never taken more than 30 seconds).
Training can be skipped by setting process=False when creating a LogisticRegression object.
In that case, weights must be provided when calling LogisticRegression.test. (set the weights parameter to weights)
There is a set of weights saved as weights in main, pass them in to test.

NaiveBayes:

Run NaiveBayes.py without any arguments.

Can run two different datasets, runs airline data by default.
Uncomment the other data and comment out the processing of the airline data in main if the other dataset should be run.

