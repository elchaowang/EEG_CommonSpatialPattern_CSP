import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from EEGSignalsClass import *
from ML_Algorithms import *
from scipy.signal import butter, lfilter

def bandpass_butter(lowcut, highcut, fs, order=4):
  nyq = 0.5*fs
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = bandpass_butter(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def loadPersonalData(IndexOfPerson, NumOfSesson = 8, nbTrialPerSesson = 20, SampleOfTrial = 5120):
    dataFile = 'C://Users/Vicon/PycharmProjects/CSP_SVM/matFiles/S0' + str(IndexOfPerson) + '.mat'
    SubData = scio.loadmat(dataFile)
    dataset = SubData['dataset']
    EEGSignalStorage = list()
    Y = np.array([])
    for sesson in range(NumOfSesson):
        trialStart =dataset[0][sesson][0][0][1][0]
        for i in range(nbTrialPerSesson):
            X = dataset[0][sesson][0][0][0][trialStart[i]-1:trialStart[i]+SampleOfTrial-1,:]
            EEGSignalStorage.append(X)
        TrainingY = dataset[0][sesson][0][0][2][0]
        Y = np.append(Y, TrainingY)
    EEGSignalStorage = np.array(EEGSignalStorage)
    samplingFrequency = dataset[0][sesson][0][0][3][0]
    # print(samplingFrequency)
    types = dataset[0][sesson][0][0][4][0]
    classes = list()
    classes.append(types[0][0])
    classes.append(types[1][0])
    eeg = EEGSignal(x=EEGSignalStorage, trial=0, y= Y, fs=samplingFrequency, classes=classes)
    return eeg

def split_dataset(eeg, rate=0.2):
    lenOfTesting = int(len(eeg.X)*rate)
    indexList = np.random.choice(len(eeg.X), lenOfTesting)
    eegTestingX = list()
    eegTestingY = np.array([])
    eegTrainingX = list()
    eegTrainingY = np.array([])

    for index in indexList:
        eegTestingX.append(eeg.X[index])
        eegTestingY = np.append(eegTestingY, eeg.y[index])
    for index in range(len(eeg.X)):
        if index not in indexList:
            eegTrainingX.append(eeg.X[index])
            eegTrainingY = np.append(eegTrainingY, eeg.y[index])
    eegTestingX = np.array(eegTestingX)
    eegTrainingX = np.array(eegTrainingX)
    TrainingEEG = EEGSignal(x=eegTrainingX, trial=0, y = eegTrainingY, fs = eeg.fs, classes = eeg.classes)
    TestingEEG = EEGSignal(x=eegTestingX, trial=0, y=eegTestingY, fs=eeg.fs, classes=eeg.classes)
    return TrainingEEG, TestingEEG

# filtering the original data
def filter_eeg(eeg, lowcut=10, highcut=30, fs=512, order=8):
    for trial in range(len(eeg.X)):
        for channel in range(len(np.transpose(eeg.X[trial]))):
            tmp = butter_bandpass_filter(np.transpose(eeg.X[trial])[channel], lowcut, highcut, fs, order)
            for sample in range(len(eeg.X[trial])):
                eeg.X[trial][sample][channel] = tmp[sample]
    return eeg

# parameters of this EEG decoding model
IndexOfPerson = 1
lowcutOfOriginalData = 10
highcutOfOriginalData = 30
filterOrders = 8
samplingFrequency = 512
nbFilterPairs = 6
numOfEstimators = 500
trainTimes = 200

eeg = loadPersonalData(IndexOfPerson=1, NumOfSesson=8, nbTrialPerSesson=20,SampleOfTrial=5120)
eeg = filter_eeg(eeg, lowcut=lowcutOfOriginalData, highcut=highcutOfOriginalData, fs=samplingFrequency,
                 order=filterOrders)

sum = list()
for i in range(200):
    # randomly select the testing dataset and training dataset
    trainEEG, testEEG = split_dataset(eeg, rate=0.375)
    TrainedCSPMatrix = learnCSP(trainEEG)
    trainDataset = extractCSPFeatures(trainEEG, TrainedCSPMatrix, nbFilterPairs=nbFilterPairs)
    testDataset = extractCSPFeatures(testEEG, TrainedCSPMatrix, nbFilterPairs= nbFilterPairs)

    X_train = trainDataset[:, 0:2*nbFilterPairs]
    Y_train = trainDataset[:, 2*nbFilterPairs]
    X_test = testDataset[:, 0:2*nbFilterPairs]
    Y_test = testDataset[:, 2*nbFilterPairs]

    accuracy = random_forest_prediction(X_train=X_train, X_test=X_test, y_train=Y_train, y_test= Y_test,
                             numOfEstimators=numOfEstimators)
    sum.append(accuracy)
    print('training time is ' + str(i) + ', and the accuracy is ' + str(accuracy))
sum = np.array(sum)
print('the average accuracy of Random Forest Classifier is : ')
print(np.sum(sum)/trainTimes)
print('the peak accuracy of Random Forest Classifier is : ')
print(np.max(sum))
print('the median accuracy of Random Forest Classifier is : ')
print(returnMedian(sum))