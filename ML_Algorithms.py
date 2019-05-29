import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

def write_excels(a, filename):
    data_pd = pd.DataFrame(a)
    file = pd.ExcelWriter(filename)
    data_pd.to_excel(file, 'sheet1', float_format='%.5f', index=False, header=None)
    file.save()

def returnMedian(x):
    x = np.sort(x)
    length = len(x)
    if length%2 == 0:
        med = (x[int(length/2)]+x[int(length/2)+1])/2
    else:
        med = x[(length+1)/2]
    return med

def random_forest_classifier(features, lables, trainingTimes=50, numOfEstimators=1000, testSize=0.2):
    sum = list()
    for i in range(trainingTimes):
        X_train, X_test, y_train, y_test = train_test_split(features, lables, test_size=testSize)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        predictModel = RandomForestClassifier(n_estimators=numOfEstimators, bootstrap=True, max_features='sqrt')

        predictModel.fit(X_train, y_train)
        rf_predictions = predictModel.predict(X_test)
        rf_probs = predictModel.predict_proba(X_test)[:, 1]
        roc_value = roc_auc_score(y_test, rf_probs)
        sum.append(roc_value)
        print('training time is ' + str(i) + ', and the accuracy is '+str(roc_value))
    sum = np.array(sum)
    print('the average accuracy of Random Forest Classifier is : ')
    print(np.sum(sum)/trainingTimes)
    print('the peak accuracy of Random Forest Classifier is : ')
    print(np.max(sum))
    print('the median accuracy of Random Forest Classifier is : ')
    print(returnMedian(sum))


def svm_classifier(features, lables, kernelMethod='linear', gam='auto'):
    # sum = 0
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(features, lables, test_size=0.375)
        # svclassifier = SVC(kernel='poly', degree=8, gamma= 'scale')
        svclassifier = SVC(kernel=kernelMethod, gamma=gam)
        svclassifier.fit(X_train, y_train)
        y_predict = svclassifier.predict(X_test)
        ac = confusion_matrix(y_test, y_predict)
        print(ac)
        print(classification_report(y_test, y_predict))
    print('the average accuracy of Support Vector Machine Classifier is : ')


def random_forest_prediction(X_train, X_test, y_train, y_test, numOfEstimators=300):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    predictModel = RandomForestClassifier(n_estimators=numOfEstimators, bootstrap=True, max_features='sqrt')

    predictModel.fit(X_train, y_train)
    rf_predictions = predictModel.predict(X_test)
    rf_probs = predictModel.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, rf_probs)
    # meanError = metrics.mean_absolute_error(y_test, rf_predictions)
    # meanSquaredError = metrics.mean_squared_error(y_test, rf_predictions)
    # RootMeanSquaredError = np.sqrt(meanSquaredError)
    accuracy = metrics.accuracy_score(y_test, rf_predictions)
    return accuracy