# 2017 Indiana Medicaid Data Challenge

Dataset: ![link](https://hub.mph.in.gov/dataset?tags=Medicaid&page=1)

Final Visualization: ![link](https://public.tableau.com/profile/jivitesh.poojary1464#!/vizhome/INMedicaidChallenge-MentalHealth/ProjectOverview?publish=yes)

Presentation: ![link](https://prezi.com/view/w8lmPrFwuUAa4oclYSyI/)

Different Tables - Categorized into bins.

## Emergency Visits 
- Zip Code and Age Group
- Race Gender and Age Group
- Provider and Age Group
- Diagnosis Code and Age group

## Medicaid Claims
- Subprogram Funding Source
- Race and Gender
- Zip Code

## Transportation Related Claims
- Race, Gender and Age
- Zip Code
- Provider
- Category of Service

## High Cost
- Provider
- Zip Code

## Hundred Most Costly Diagnostic Claims

## CLAIMS SERVICING DIABETES PATIENTS
- Race and Gender
- Provider
                    
## CLAIMS SERVICING MENTAL HEALTH PATIENT
- Zip Code
- Race AND Gender
- Provider

## Patient Population
- Provider Type
- Provider Speciality

--------
## Task0

### Prerequisite

Since the codes are written in Python 3, if you run the code on the campus servers, turn Python 3 module on by running:

```
module load python/3.6.0
```

- Parameter tuning for K nearest neighbor

    - number of neighbors: (3, 5, 7, ... , 19)
    - weights tuple: ("uniform", "distance")

- Parameter tuning for Random Forest

    - criterion tuple: ("gini", "entropy")
    - number of trees: 4, 8, 16, ... , 4096
    - minimum number of samples required to split an internal node: 2, 4, 8, ... , 32

- Parameter tuning for SVM

    - kernel: 'linear', 'poly', 'rbf', 'sigmoid'
    - penalty parameter C of the error term: 2^(-3), 2^(-1), ..., 2^(15)
    - gamma (Kernel coefficient for 'rbf', 'poly' and 'sigmoid'): 2^(-15), 2^(-13), ..., 2^(3)
    - degree of the polynomial kernel function: 1, 2, ... , 4

- Parameter tuning for Neural Network

    - hidden layers: 27 different hidden layers (16, 16, 16) to (64, 64, 64)
    - activation function for the hidden layer: 
        - identity: f(x) = x
        - logistic: f(x) = 1 / (1 + exp(-x))
        - tanh: f(x) = tanh(x)
        - relu: f(x) = max(0, x)
    - solver for weight optimization:
        - 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Ji    mmy

### Best Parameters I found

0.949,criterion=gini,n=8,minss=8,RandomForest
0.789,n=17,weights=uniform,kNN
0.738,hls=(16, 16, 16),alpha=0.0625,activation=identity,solver=adam,NeuralNetwork
0.738,kernel=rbf,C=0.125,gamma=0.03125,SVM

## Experiment1 (6 features only) 95%

K-fold (K = 5)
```
python Kfold.py -i Additional\ Data/MHA-Model-Data-0.csv -o MHA_model0
```

Parameter Sweeping for 4 classifiers
```
python main_rf_parameter_sweep.py -i MHA_model0.npz
```

```
python main_rf.py -i MHA_model0.npz -criterion gini -n 8 -minss 8
```

## Experiment2 (original + additional features) 91%

K-fold (K = 5)
```
python Kfold.py -i Additional\ Data/MHA-Model-Data-1.csv -o MHA_model2
```

Parameter Sweeping for 4 classifiers
```
python main_rf_parameter_sweep.py -i MHA_model2.npz
```

## Experiment3 (remove 4 features - remain provider_count mhps_present) 90%

K-fold (K = 5)
```
python Kfold.py -i Additional\ Data/MHA-Model-Data-3.csv -o MHA_model3
```

Parameter Sweeping for 4 classifiers
```
python main_rf_parameter_sweep.py -i MHA_model3.npz
```
## Experiment4 (remove Number of providers and Mental health...) 86%

K-fold (K = 5)
```
python Kfold.py -i Additional\ Data/MHA-Model-Data-4.csv -o MHA_model4
```

Parameter Sweeping for 4 classifiers
```
python main_rf_parameter_sweep.py -i MHA_model4.npz
```
--------
--------

## Task1: Find the impact of the Obamacare using Classification (Random Forest)

### Data Description

Before processing
```
>>> df.shape
(3681, 6)
```

After processing (Categorical values into 0,1)
```
>>> df2.shape
(3681, 95)
```

### Data Cleaning

The entries in 2012 ~ 2014 period: set to 0

The entries in 2015 ~ 2016 period: set to 1

From the categorical feature, create different columns with the number of different category, and then set it to 1 if the entry is the category. Else 0.
```
python -i categorical_to_01.py data_processed/high_cost_claims_by_zip_code.csv
```

From the dataset, use K-fold for training and testing the model. I set k = 5.
```
python Kfold.py -i data_processed/to_dummies.csv -o to_dummy
```

The following command sweeps through parameters to find the best parameter to explain the dataset.
```
python main_rf_parameter_sweep.py -i to_dummy.npz
```

Check the accuracy using the following command. (Random forest, gini, n=32, min splits=32)
```
python main_rf.py -i to_dummy.npz -criterion gini -n 32 -minss 32
```

### Best classifier and its corresponding parameter

- Random Forest
- gini index
- n = 32
- min splits = 32

```
0.791,criterion=gini,n=32,minss=32,RandomForest
```



## Incorporate race information in diabete data 

### Data Cleaning

- `diabete.py`

Input files: `claims_servicing_diabetes_patients_by_recipient_location.csv` and `zipcode_race.csv`
Output files: five `.csv` files, each containing the patients grouped by race in each zip code

Usage

```
$ python diabetes.py data/claims_servicing_diabetes_patients_by_recipient_location.csv data_additional/zipcode_race.csv
```
