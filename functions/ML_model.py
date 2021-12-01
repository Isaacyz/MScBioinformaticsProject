import math
import csv
import decimal
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import recall_score, make_scorer

def float_range(start, stop, step, round_n):
  while start < stop:
    yield round(start,round_n)
    start += float(decimal.Decimal(step))

def acc(TP, FP, TN, FN):
    # Cacluate accuracise
    accuracy_0 = TN/(TN+FN)
    accuracy_1 = TP/(TP+FP)
    accuracy_all = (TN+TP)/(TN+FN+TP+FP)
    return [accuracy_all,accuracy_0, accuracy_1]

def MCC(TP, FP, TN, FN):
    # Cacluate Matthews Correlation Coefficient
    return (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

def Sensitivity(TP, FN):
    # Cacluate sensitivity
    return TP/(TP+FN)
def Specificty(FP, TN):
    # Cacluate Specificty
    return TN/(TN+FP)

def run_calculation(x, TP, FP, TN, FN):
    if x == "accuracies":
        return acc(TP, FP, TN, FN)
    if x == "MCC":
        return MCC(TP, FP, TN, FN)
    if x == "sensitivity":
        return Sensitivity(TP, FN)
    if x == "specificty":
        return Specificty(TP, FN)

def Prediction_Performance(validation,predictions, calculations=[]):
    TP, FP = 0,0
    TN, FN = 0,0
    one = sum(validation)
    all_len = len(validation)
    for m, n in zip(validation, predictions):
        if m == n == 0:
            TN += 1
        if m == n == 1:
            TP += 1
    FP = one - TP
    FN = all_len - one - TN

    output = {}
    while calculations:
        x = calculations.pop()
        output[x]= run_calculation(x, TP, FP, TN, FN)
    return output


# The following function is the workflow on given features to build a ML predictor
# The input is training dataset with given number of features
# LASSO regression further removes some features by given penalty values
# Build a ML predictor based on the balanced training samples
# Evaluate model performance based on the validation datasets.
def ML_predictor_building(X_lasso, y_lasso, data_train_balanced, penality_list ,file, Model, data_validation=[]):
    output=[]
    for a in penality_list:
        print(a)
        reg = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=1/a)
        reg.fit(X_lasso.values, y_lasso.values)
        reg_non_zero = [True if abs(i) > 0 else False for i in reg.coef_[0]]
        names = [X_lasso.columns[i] for i in range(0, len(reg_non_zero)) if reg_non_zero[i]]
        # If the penalty value ends up with <20 features, coutinous to the nest loop
        if len(names) < 20:
            continue

        # Use ML to build predictor
        X1 = data_train_balanced.loc[:, names]
        y = data_train_balanced.loc[:,"Response"]
        if Model == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif Model == 'RF':
            model = RandomForestClassifier()
        else:
            model = svm.SVC(kernel="linear")
        # define 'k' in k-fold Cross Validation
        condition1 = len(data_train_balanced[data_train_balanced["Response"] == 1].index)
        condition2 = len(data_train_balanced[data_train_balanced["Response"] == 0].index)
        if condition1 < 5 or condition2 < 5:
            n = 3
        elif condition1 >= 10 and condition2 >= 10:
            n = 10
        else:
            n = 5
        # The performance on predicting Training dataset is represented by following CV accuracies
        cv = RepeatedStratifiedKFold(n_splits=n, n_repeats=10, random_state=1)
        scoring = {'recall0': make_scorer(recall_score, average = None, labels = [0]),
                    'recall1': make_scorer(recall_score, average = None, labels = [1])}
        scores = cross_validate(model, X1, y, scoring = scoring, cv = cv, return_train_score = False)
        avg_score =[]
        print(sum(y.tolist()))
        print(len(y.tolist()))
        for i in range(len(scores['test_recall1'])):
            avg_score.append((scores['test_recall1'][i]*sum(y.tolist()) + scores['test_recall0'][i]*(len(y.tolist())-sum(y.tolist())))/len(y.tolist()))

        # The performance on predicting Testing datasets are represented by accuracies/MCC scores
        s = []
        for testing in data_validation:
            model.fit(X1, y)
            y_validation = testing['Response'].tolist()
            predictions = model.predict(testing.loc[:, names])
            result_of_testing = Prediction_Performance(y_validation,predictions,["accuracies","MCC"])
            s.append(result_of_testing)

        output.append([a, s, mean(avg_score), std(avg_score), mean(scores['test_recall0'].tolist()),
                        std(scores['test_recall0'].tolist()), mean(scores['test_recall1'].tolist()),
                        std(scores['test_recall1'].tolist()), len(names), "/".join(names)])

        # Store in given file
        with open('{}.csv'.format(file), 'a') as f:
            write = csv.writer(f, delimiter=",")
            for j in output:
                write.writerow(j)
