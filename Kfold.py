import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.model_selection
import sklearn.metrics
from sklearn.decomposition import PCA

def get_X_y(infile):
    X = []
    y = []
    infile.readline()
    for line in infile:
        line = line.rstrip().split(',')
        x = line[1:-1]
        X.append([int(value) for value in x])
        y.append(int(line[-1]))

    return np.array(X), np.array(y)

def prepare_train_test_using_kfold(k, X, y):
    trainset, testset = [], []
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    print(kf)
    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="Use expressionmRNAAnnotations.txt", default=sys.stdin)
    parser.add_argument('-o', '--outfile', help="output file name")
    parser.add_argument('-kfold', '--kfold', type=int, default=5, help="k-fold (positive integer. default is 5-fold)")
    args = parser.parse_args()

    k_fold = args.kfold
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X and y from the input txt file...")
    X, y = get_X_y(args.infile)
    
    #####################################################################
    # Get trainsets and testsets using K-Fold
    print("Preparing the trainsets and testsets using {} fold...".format(k_fold))
    X_train, y_train, X_test, y_test = prepare_train_test_using_kfold(k_fold, X, y)

    np.savez(args.outfile, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, Kfold = k_fold)
    print("-"*60)
    print("trainset and testset are created!")
