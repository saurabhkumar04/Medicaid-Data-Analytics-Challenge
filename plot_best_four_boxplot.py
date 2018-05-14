import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from argparse import ArgumentParser

n = 4

def read_values(input_file, kfold):
    X = np.zeros((n, kfold))
    label = []
    for i in range(n):
        for k in range(kfold):
            line = input_file.readline().rstrip().split(',')
            X[i][k] = float(line[0])
        label.append(line[-1])
    return X, label

def plot_accuracy(X, label, title, xlabel, ylabel, filename):
    plt.boxplot(X.T, labels=label)
    # index = np.arange(X.shape[0])
    # plt.bar(index, X, align="center")
    # plt.xticks(index, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(X.min()-0.05, X.max()+0.05)
    plt.ylim(0.6, 1)
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description="Read the best results, then draw a line curve.")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="input file", default=sys.stdin)
    parser.add_argument('-k', '--kfold', type=int, help="kfold", default=5)
    parser.add_argument('-t', '--title', type=str, help="title of the plot")
    parser.add_argument('-o', '--outfile', type=str, help="output file")
    args = parser.parse_args()

    data = args.infile
    kfold = args.kfold
    X, label = read_values(data, kfold)
    plot_accuracy(X, label, "Model3: Predicting using only socio-economic features", "Classifier type", "Accuracy", args.outfile)
