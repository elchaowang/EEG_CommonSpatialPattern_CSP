import numpy as np
import pandas as pd
from ML_Algorithms import *
class EEGSignal:
    X = None
    trial = None
    y = None
    fs = 512
    classes = list()

    def __init__(self,x,trial,y,fs,classes):
        self.X = x
        self.trial = trial
        self.y = y
        self.fs = fs
        self.classes = classes

def learnCSP(eeg):
    nbChannels = len(eeg.X[0][0])
    nbTrials = len(eeg.X)
    classLables = np.unique(eeg.y)
    nbClasses = len(classLables)

    if nbClasses!=2:
        print('ERROR! CSP can only be used for two classes!')
        return
    covMatrices = [np.array([]), np.array([])]
    trialCov = np.zeros((nbTrials, nbChannels, nbChannels))
    for trialNum in range(nbTrials):
        E = eeg.X[trialNum,:,:].transpose()
        E_ = E.transpose()
        EE = np.dot(E, E_)
        trialCov[trialNum,:,:] = EE/np.trace(EE)
    del E
    del EE

    for c in range(nbClasses):
        classes = list()
        for i in range(nbTrials):
            if eeg.y[i] == classLables[c]:
                classes.append(trialCov[i])
        classes = np.array(classes)
        covMatrices[c] = np.mean(classes,0)
    covMatrices = np.array(covMatrices)
    covTotal = covMatrices[0] + covMatrices[1]

############## whitening transform of total covariance matrix ###########################
    eigenvalues, Ut = np.linalg.eig(covTotal)
    egIndex = np.argsort(-eigenvalues)
    eigenvalues = sortEigs(egIndex, eigenvalues)
    Ut = sortVectorByEigs(egIndex, Ut)

########### transforming covariance matrix of first class using P ########################
    P = np.dot(np.diag(np.sqrt(1.0/eigenvalues)), np.transpose(Ut))
    tmp = np.dot(covMatrices[0], np.transpose(P))
    transformedCov1 = np.dot(P, tmp)
    filename = 'transformedCov1.xlsx'
    write_excels(np.around(transformedCov1, decimals=4), filename)

################## EVD of the transformed covariance matrix ##############################
    eigenvalues, U1 = np.linalg.eig(transformedCov1)
    egIndex = np.argsort(-eigenvalues)
    U1 = sortVectorByEigs(egIndex, U1)
    CSPMatrix = np.dot(np.transpose(U1), P)
    return CSPMatrix

def extractCSPFeatures(EEGSignals, CSPMatrix, nbFilterPairs):
    Filter = np.zeros((2*nbFilterPairs, len(CSPMatrix)))
    nbTrials = len(EEGSignals.X)
    features = np.zeros((nbTrials, 2*nbFilterPairs+1))
    for index in range(nbFilterPairs):
        Filter[index] = CSPMatrix[index]
        Filter[2*nbFilterPairs-1-index] = CSPMatrix[len(CSPMatrix)-1-index]

    for t in range(nbTrials):
        projectedTrial = np.dot(Filter, np.transpose(EEGSignals.X[t, :, :]))
        # print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        # print(projectedTrial)
        variances = np.var(projectedTrial, ddof=1, axis=1)
        # print(variances)
        for f in range(len(variances)):
            features[t, f] = np.log(variances[f])
        features[t, 2*nbFilterPairs] = EEGSignals.y[t]

    return features
    # projectedTrial = np.dot(Filter, np.transpose(EEGSignals.X[0, :, :]))
    # print(len(projectedTrial))

    # variances =

def sortEigs(egIndex, eigenvalues):
    tmp = list()
    for i in range(len(eigenvalues)):
        tmp.append(eigenvalues[i])
    tmp = np.array(tmp)
    for i in range(len(eigenvalues)):
        eigenvalues[i] = tmp[egIndex[i]]

    return eigenvalues

def sortVectorByEigs(egIndex, Ut):
    tmp = list()
    for i in egIndex:
        temp = Ut[:, egIndex]
        tmp.append(temp)
    Ut = np.array(tmp[0])
    # for i in range(len(Ut)):
    #     Ut[i][2] = -Ut[i][2]
    #     Ut[i][7] = -Ut[i][7]
    #     Ut[i][8] = -Ut[i][8]
    #     Ut[i][10] = -Ut[i][10]
    #     Ut[i][11] = -Ut[i][11]
    #     Ut[i][13] = -Ut[i][13]
    #     Ut[i][14] = -Ut[i][14]

    return Ut