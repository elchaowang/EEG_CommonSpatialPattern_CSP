import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ML_Algorithms import *

# trainingData = pd.read_excel('CSPTrainingData.xlsx')
# X_train = trainingData.iloc[:,0:6]
# y_train = trainingData.iloc[:,6]
# testingData = pd.read_excel('CSPTestingData.xlsx')
# X_test = testingData.iloc[:,0:6]
# y_test = testingData.iloc[:,6]

# open the data file which contains the features and lables
datafile = open('P1Data\/P1TotalData10_30Hz.xlsx', 'rb')
# read the data file using pandas, and store the feature matrix as DataFrame
trainingData = pd.read_excel(datafile)

datafile.close()
# get the number of features
lenOfFeatures = len(trainingData.iloc[1,:]) - 1

print('The program has found '+ str(lenOfFeatures) + ' features')

# features is X, lables is Y
features = trainingData.iloc[:,0:lenOfFeatures]
lables = trainingData.iloc[:,lenOfFeatures]

# apply the RF classifier and print out the accuracy
random_forest_classifier(features, lables, trainingTimes=50, numOfEstimators=1000, testSize=0.375)
# svm_classifier(features, lables, kernelMethod='poly', gam='scale')