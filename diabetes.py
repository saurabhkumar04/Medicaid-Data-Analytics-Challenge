# Reads two files, and creates 5 csv files.

from __future__ import division
import sys
import pandas as pd
from pandas import DataFrame

def read_diabete(filename):
    zipcode = {}
    file = open(filename, "r")
    # remove header
    file.readline()
    for line in file:
        line = line.rstrip().split(',')
        if int(line[1]) not in zipcode:
            zipcode[int(line[1])] = []
            zipcode[int(line[1])].append((int(line[0]),int(line[2])))
        else:
            zipcode[int(line[1])].append((int(line[0]),int(line[2])))
    return zipcode

def read_race(filename):
    zipcode = {}
    file = open(filename, "r")
    # remove first line
    file.readline()
    # get header
    header = file.readline()
    header = header.rstrip().split(',')
    # modify header
    header[3] = "Total"
    header[5] = "White"
    header[7] = "Black and African American"
    header[9] = "American Indian and Alaska Native"
    header[11] = "Asian"
    header[13] = "Native Hawaiian and Other Pacific Islander"
    header[15] = "Some other race"
    header[17] = "Two or more races"
    header[19] = "Two races including Some other race"
    header[21] = "Two races excluding Some other race and three or more races"

    state = file.readline()
    # Starting reading zipcodes...
    for line in file:
        line = line.rstrip().split(',')
        code = int(line[1])
        total = int(line[3])
        if total == 0:
            continue
        zipcode[code] = {}
        zipcode[code][header[3]] = total
        for i in range(5, 23, 2):
            zipcode[code][header[i]] = int(line[i]) / total
    return zipcode

def combine_dictionary(diabete, race):
    dict_2012 = {}
    dict_2013 = {}
    dict_2014 = {}
    dict_2015 = {}
    dict_2016 = {}
    for zipcode in diabete:
        if zipcode in race:
            dict_2012[zipcode] = {}
            dict_2013[zipcode] = {}
            dict_2014[zipcode] = {}
            dict_2015[zipcode] = {}
            dict_2016[zipcode] = {}
            for year, patient in diabete[zipcode]:
                for r in race[zipcode]:
                    if year == 2012:
                        dict_2012[zipcode][r] = int(round(patient * race[zipcode][r]))
                    if year == 2013:
                        dict_2013[zipcode][r] = int(round(patient * race[zipcode][r]))
                    if year == 2014:
                        dict_2014[zipcode][r] = int(round(patient * race[zipcode][r]))
                    if year == 2015:
                        dict_2015[zipcode][r] = int(round(patient * race[zipcode][r]))
                    if year == 2016:
                        dict_2016[zipcode][r] = int(round(patient * race[zipcode][r]))
    return dict_2012, dict_2013, dict_2014, dict_2015, dict_2016

file_diabete = sys.argv[1]
file_race = sys.argv[2]
diabete = read_diabete(file_diabete)
race = read_race(file_race)
zip_2012, zip_2013, zip_2014, zip_2015, zip_2016 = combine_dictionary(diabete, race)

df_2012 = DataFrame.from_dict(zip_2012).T
df_2013 = DataFrame.from_dict(zip_2013).T
df_2014 = DataFrame.from_dict(zip_2014).T
df_2015 = DataFrame.from_dict(zip_2015).T
df_2016 = DataFrame.from_dict(zip_2016).T

del df_2012["Total"]
del df_2013["Total"]
del df_2014["Total"]
del df_2015["Total"]
del df_2016["Total"]

df_2012.to_csv("data_processed/diabete_zipcode_race_2012.csv")
df_2013.to_csv("data_processed/diabete_zipcode_race_2013.csv")
df_2014.to_csv("data_processed/diabete_zipcode_race_2014.csv")
df_2015.to_csv("data_processed/diabete_zipcode_race_2015.csv")
df_2016.to_csv("data_processed/diabete_zipcode_race_2016.csv")
