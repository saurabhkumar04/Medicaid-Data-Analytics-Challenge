import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import classification_algo
import sklearn.metrics

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    parser.add_argument('-C', '--C', type=float,
             help="Penalty parameter C of the error term.")
    parser.add_argument('-kernel', '--kernel', type=str,
             help="Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)")
    parser.add_argument('-degree', '--degree', type=int,
             help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.")
    parser.add_argument('-gamma', '--gamma', type=float,
             help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.")
    args = parser.parse_args()

    # Load the dataset
    dataset = np.load(args.infile)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    k_fold = dataset["Kfold"]

    kernel = args.kernel
    C = args.C
    gamma = args.gamma
    degree = args.degree
    
    if kernel == 'poly':
        accuracy_list = np.zeros(k_fold)
        for i in range(k_fold):
            pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, degree, gamma)
            accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
            accuracy_list[i] = accuracy
        print("{0:.3f},kernel={1},C={2},gamma={3},degree={4},SVM".format(accuracy_list.mean(),kernel,C,gamma,degree))

    elif kernel == 'linear':
        accuracy_list = np.zeros(k_fold)
        for i in range(k_fold):
            pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, None)
            accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
            accuracy_list[i] = accuracy
        print("{0:.3f},kernel={1},C={2},SVM".format(accuracy_list.mean(),kernel,C))
    else:
        accuracy_list = np.zeros(k_fold)
        for i in range(k_fold):
            pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, gamma)
            accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
            accuracy_list[i] = accuracy
        print("{0:.3f},kernel={1},C={2},gamma={3},SVM".format(accuracy_list.mean(),kernel,C,gamma))

