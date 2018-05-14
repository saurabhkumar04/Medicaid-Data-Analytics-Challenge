import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import classification_algo
import sklearn.metrics

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    #  parser.add_argument('-C', '--C', type=float,
            #  help="Penalty parameter C of the error term.")
    parser.add_argument('-kernel', '--kernel', type=str,
             help="Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)")
    #  parser.add_argument('-degree', '--degree', type=int,
            #  help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.")
    #  parser.add_argument('-gamma', '--gamma', type=float,
            #  help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.")
    args = parser.parse_args()

    #  print("\nAfter run, the program will print accuracy of the train and test set")
    print('-'*60)
    print("Loading the dataset...")
    # Load the dataset
    dataset = np.load(args.infile)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    k_fold = dataset["Kfold"]

    #C = 2^(-3), 2^(-1), ..., 2^(15)
    #gamma = 2^(-15), 2^(-13), ..., 2^(3)
    # kernel_tuple = ('linear', 'poly', 'rbf', 'sigmoid')
    kernel = args.kernel
    C_list = np.power(2.0, np.arange(-3, 17, 2))
    gamma_list = np.power(2.0, np.arange(-15, 5, 2))
    degree_list = np.arange(1, 5)
    
    print('-'*60)
    print("SVM")
    if kernel == 'poly':
        for C in C_list:
            for gamma in gamma_list:
                for degree in degree_list:
                    accuracy_list = np.zeros(k_fold)
                    for i in range(k_fold):
                        pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, degree, gamma)
                        accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                        #  print("Run: {}, {}".format(i+1, accuracy))
                        accuracy_list[i] = accuracy
                    print("{0:.3f},kernel={1},C={2},gamma={3},degree={4},SVM".format(accuracy_list.mean(),kernel,C,gamma,degree))
                    #  print("\nAverage accuracy for SVM classifier with "  + kernel + " kernel: {0:.3f}".format(accuracy.mean()))

    elif kernel == 'linear':
        for C in C_list:
            accuracy_list = np.zeros(k_fold)
            for i in range(k_fold):
                pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, None)
                accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                #  print("Run: {}, {}".format(i+1, accuracy))
                accuracy_list[i] = accuracy
            print("{0:.3f},kernel={1},C={2},SVM".format(accuracy_list.mean(),kernel,C))
            #  print("\nAverage accuracy for SVM classifier with "  + kernel + " kernel: {0:.3f}".format(accuracy.mean()))
    else:
        for C in C_list:
            for gamma in gamma_list:
                accuracy_list = np.zeros(k_fold)
                for i in range(k_fold):
                    pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, gamma)
                    accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                    #  print("Run: {}, {}".format(i+1, accuracy))
                    accuracy_list[i] = accuracy
                print("{0:.3f},kernel={1},C={2},gamma={3},SVM".format(accuracy_list.mean(),kernel,C,gamma))

