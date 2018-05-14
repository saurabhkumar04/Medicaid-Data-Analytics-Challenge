import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.metrics
import classification_algo

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    parser.add_argument('-n', '--n', type=int,
             help="The number of trees in the forest.")
    parser.add_argument('-criterion', '--criterion', type=str,
             help="The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain. Note: this parameter is tree-specific.")
    parser.add_argument('-minss', '--minss', type=int,
             help="The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.")
    args = parser.parse_args()

    # Load the dataset
    dataset = np.load(args.infile)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    k_fold = dataset["Kfold"]

    criterion = args.criterion
    n = args.n
    minss = args.minss

    #####################################################################
    # Random Forest
    for i in range(k_fold):
        pred = classification_algo.random_forest(X_train[i], y_train[i], X_test[i], n, criterion, minss)
        accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
        print("{0:.3f},criterion={1},n={2},minss={3},RandomForest".format(accuracy,criterion,n,minss))
