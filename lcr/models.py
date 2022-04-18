""" Contains model code using daily_compress_df.csv and
monthly_compress_df.csv (see create_dataframe.py) (using levels as target for classification) """

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import lcr_global_vars
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import statistics

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np

all_vars = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC",
            "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC",
            "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF",
            "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850",
            "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010", "U200", "U500", "U850", "VBOT",
            "WSPDSRFAV", "Z050", "Z500"]
train_vars = ["bc_a1_SRF", "dst_a1_SRF", "FLNS", "FLNSC", "pom_a1_SRF", "PRECL", "PRECSC",
              "PRECSL", "Q200", "QBOT", "SHFLX", "so4_a1_SRF", "so4_a2_SRF", "so4_a3_SRF",
              "TAUX", "TAUY", "TMQ", "T010", "T200", "T500", "T850", "TREFHT",  "U200", "U500",
              "VBOT", "PSL", "FLUT"]
validate_vars = ["ICEFRAC", "LHFLX", "PRECT", "Q500", "TREFHTMN", "TS", "U850", "WSPDSRFAV", "Z500",
                 "FSNSC"]
test_vars = ["dst_a3_SRF",  "FSNS", "FSNTOA", "Q850", "TREFHTMX", "Z050", "U010", "PRECTMX"]


def random_forest(X_train, X_test, y_train, y_test):
    params = {
        "max_depth": [2, 5],
        "random_state": [0]
    }
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring="accuracy", cv=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return (y_pred, accuracy)

def adaboost(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": [50],
        "learning_rate": [0.1]
    }
    clf = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=params, scoring="accuracy", cv=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return (y_pred, accuracy)

def neural_net(X_train, X_test, y_train, y_test):
    X_train_arr = X_train.to_numpy().astype(float)
    X_test_arr = X_test.to_numpy().astype(float)

    layers = [Dense(10, activation='relu', input_shape=(3,)),
              Dense(10, activation='relu'),
              Dense(max(y_train)+1)]
    model = keras.Sequential(layers=layers)
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train_arr, y_train, epochs=50)
    (test_loss, test_acc) = model.evaluate(X_test_arr, y_test)
    return(np.argmax(model.predict(X_test_arr), axis=-1), test_acc)

def kNN(X_train, X_test, y_train, y_test):
    params = {
        "n_neighbors": [1, 2, 3, 4, 5, 10, 20, 50, 100]
    }
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, scoring="accuracy", cv=10)

    clf.fit(X_train, y_train)
    # Step 3 - Predict the validation data
    testPredictions = clf.predict(X_test)
    acc = accuracy_score(y_test, testPredictions)
    return(testPredictions, acc, clf.best_params_["n_neighbors"])

def SVM(X_train, X_test, y_train, y_test):
    params = {
        "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4],
        "random_state": [0],
        "gamma": ["auto"]
    }
    clf = GridSearchCV(estimator=SVC(), param_grid=params, scoring="accuracy", cv=10)

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    return(y_pred, acc)

def LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test):
    params = {
        "n_components": [1],
    }
    clf = GridSearchCV(estimator=LDA(), param_grid=params, scoring="accuracy", cv=10)

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    return(y_pred, acc)

def QuadraticDiscriminantAnalysis(X_train, X_test, y_train, y_test):
    params = {
        # "n_components": [1],
    }
    clf = GridSearchCV(estimator=QDA(), param_grid=params, scoring="accuracy", cv=10)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    return(y_pred, acc)


def PredMostFrequent(X_train, X_test, y_train, y_test):
    (rf_preds, rf_acc) = random_forest(X_train, X_test, y_train, y_test)
    (nn_preds, nn_acc) = neural_net(X_train, X_test, y_train, y_test)
    (knn_preds, knn_acc, knn_params) = kNN(X_train, X_test, y_train, y_test)
    (svm_preds, svm_acc) = SVM(X_train, X_test, y_train, y_test)
    (lda_preds, lda_acc) = LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    (qda_preds, qda_acc) = QuadraticDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    y_pred = []
    for i in range(0,len(y_test)):
        y_pred.append(statistics.mode([rf_preds[i],
                                       nn_preds[i],
                                       knn_preds[i],
                                       svm_preds[i],
                                       lda_preds[i],
                                       qda_preds[i]]))

    return(y_pred, sum(np.array(y_pred)==np.array(y_test)) / len(np.array(y_test)))


if __name__ == "__main__":
    daily_df = pd.read_csv('../data/zfp_only/daily_compress_df.csv')
    # monthly_df = pd.read_csv('../data/monthly_compress_df.csv')

    # just look at a particular algorithm and try and guess the level for now
    subset_daily = daily_df[daily_df["algs"] == "zfp"]
    X = subset_daily[lcr_global_vars.features]
    y = subset_daily[["levels"]]
    y = np.array(y).ravel()
    y = np.unique(y, return_inverse=True)[1]

    # create train-test split at random
    # X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                     y,
    #                                                     test_size = 0.33, random_state = 42)

    # create train-test split by selecting variables
    X_train = X[subset_daily["variable"].isin(train_vars)]
    X_validate = X[subset_daily["variable"].isin(validate_vars)]
    X_test = X[subset_daily["variable"].isin(test_vars)]

    y_train = y[subset_daily["variable"].isin(train_vars)]
    y_validate = y[subset_daily["variable"].isin(validate_vars)]
    y_test = y[subset_daily["variable"].isin(test_vars)]

    # (rf_preds, rf_acc) = random_forest(X_train, X_test, y_train, y_test)
    # print(rf_acc)
    # print(confusion_matrix(y_test, rf_preds))
    # report = classification_report(y_test, rf_preds, output_dict=True)
    # rf_df = pd.DataFrame(report).transpose()
    # rf_df.to_csv('../data/rf_report.csv', float_format="%.3f")

    (boost_preds, boost_acc) = adaboost(X_train, X_test, y_train, y_test)
    print(boost_acc)
    print(confusion_matrix(y_test, boost_preds))
    report = classification_report(y_test, boost_preds, output_dict=True)
    boost_df = pd.DataFrame(report).transpose()
    boost_df.to_csv('../data/boost_report.csv', float_format="%.3f")

    # (nn_preds, nn_acc) = neural_net(X_train, X_test, y_train, y_test)
    # print(nn_acc)
    # print(confusion_matrix(y_test, nn_preds))
    # report = classification_report(y_test, nn_preds, output_dict=True)
    # nn_df = pd.DataFrame(report).transpose()
    # nn_df.to_csv('../data/nn_report.csv', float_format="%.3f")
    #
    # (knn_preds, knn_acc, knn_params) = kNN(X_train, X_test, y_train, y_test)
    # print(knn_acc)
    # print(confusion_matrix(y_test, knn_preds))
    # report = classification_report(y_test, knn_preds, output_dict=True)
    # knn_df = pd.DataFrame(report).transpose()
    # knn_df.to_csv('../data/knn_report.csv', float_format="%.3f")
    #
    # (svm_preds, svm_acc) = SVM(X_train, X_test, y_train, y_test)
    # print(svm_acc)
    # print(confusion_matrix(y_test, svm_preds))
    # report = classification_report(y_test, svm_preds, output_dict=True)
    # svm_df = pd.DataFrame(report).transpose()
    # svm_df.to_csv('../data/svm_report.csv', float_format="%.3f")
    #
    # (lda_preds, lda_acc) = LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    # print(lda_acc)
    # print(confusion_matrix(y_test, lda_preds))
    # report = classification_report(y_test, lda_preds, output_dict=True)
    # lda_df = pd.DataFrame(report).transpose()
    # lda_df.to_csv('../data/lda_report.csv', float_format="%.3f")
    #
    # (qda_preds, qda_acc) = QuadraticDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    # print(qda_acc)
    # print(confusion_matrix(y_test, qda_preds))
    # report = classification_report(y_test, qda_preds, output_dict=True)
    # qda_df = pd.DataFrame(report).transpose()
    # qda_df.to_csv('../data/qda_report.csv', float_format="%.3f")
    #
    # (combine_preds, combine_acc) = PredMostFrequent(X_train, X_test, y_train, y_test)
    # print(combine_acc)
    # print(confusion_matrix(y_test, combine_preds))
    # report = classification_report(y_test, combine_preds, output_dict=True)
    # combine_df = pd.DataFrame(report).transpose()
    # combine_df.to_csv('../data/combine_report.csv', float_format="%.3f")