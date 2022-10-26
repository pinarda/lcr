""" Contains model code using daily_compress_df.csv and
monthly_compress_df.csv (see create_dataframe.py) (using levels as target for classification) """

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

import os
import argparse
import matplotlib.pyplot as plt
import sys
import pickle
import inspect
import re
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import data_gathering.lcr_global_vars as lcr_global_vars
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import statistics

from tensorflow.keras.layers import Dense

import numpy as np

all_vars = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC",
            "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "LHFLX", "pom_a1_SRF", "PRECL", "PRECSC",
            "PRECSL", "PRECT", "PRECTMX", "PSL", "Q200", "Q500", "Q850", "QBOT", "SHFLX", "so4_a1_SRF",
            "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850",
            "TAUX", "TAUY", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TS", "U010", "U200", "U500", "U850",
            "UBOT", "V200", "V500", "V850", "VBOT",
            "WSPDSRFAV", "Z050", "Z500"]
train_vars = ["bc_a1_SRF", "dst_a1_SRF", "dst_a3_SRF", "FLNS", "FLNSC",
            "FLUT", "FSNS", "FSNSC", "FSNTOA", "ICEFRAC", "pom_a1_SRF", "PRECL", "PRECSC",
            "PRECSL", "PRECT", "PRECTMX", "Q200", "Q500", "Q850", "so4_a1_SRF",
            "so4_a2_SRF", "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "T010", "T200", "T500", "T850",
              "TREFHT", "TREFHTMN", "TREFHTMX", "U010", "U200", "U500", "U850"]
test_vars = ["LHFLX", "QBOT", "TAUX", "TMQ", "UBOT", "V850", "VBOT", "WSPDSRFAV"]
validate_vars = ["PSL", "SHFLX", "TAUY", "TS", "V200", "V500", "Z050", "Z500"]

monthly_vars = ["ABSORB", "ANRAIN", "ANSNOW", "AODABS", "AODDUST1",
                "AODDUST2", "AODDUST3", "AODVIS", "AQRAIN", "AQSNOW",
                "AREI", "AREL", "AWNC", "AWNI", "bc_a1_SRF", "BURDENBC",
                "BURDENDUST", "BURDENPOM", "BURDENSEASALT", "BURDENSO4",
                "BURDENSOA", "CCN3", "CDNUMC", "CLDHGH", "CLDICE",
                "CLDLIQ", "CLDLOW", "CLDMED", "CLDTOT", "CLOUD",
                "CO2", "CO2_FFF", "CO2_LND", "CO2_OCN", "DCQ",
                "dst_a1_SRF", "dst_a3_SRF", "DTCOND", "DTV", "EXTINCT",
                "FICE", "FLDS", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLUT",
                "FLUTC", "FREQI", "FREQL", "FREQR", "FREQS", "FSDS", "FSDSC",
                "FSNS", "FSNSC", "FSNT", "FSNTC", "FSNTOA", "FSNTOAC",
                "ICEFRAC", "ICIMR", "ICLDIWP", "ICLDTWP", "ICWMR", "IWC",
                "LANDFRAC", "LHFLX", "LWCF", "NUMICE", "NUMLIQ", "OCNFRAC",
                "OMEGA", "OMEGAT", "PBLH", "PHIS", "pom_a1_SRF", "PRECC",
                "PRECL", "PRECSC", "PRECSL", "PS", "PSL", "Q", "QFLX", "QRL",
                "QRS", "RELHUM", "SFCO2", "SFCO2_FFF", "SFCO2_LND", "SFCO2_OCN",
                "SHFLX", "SNOWHICE", "SNOWHLND", "so4_a1_SRF", "so4_a2_SRF",
                "so4_a3_SRF", "soa_a1_SRF", "soa_a2_SRF", "SOLIN", "SRFRAD",
                "SWCF", "T", "TAUX", "TAUY", "TGCLDIWP", "TGCLDLWP", "TMCO2",
                "TMCO2_FFF", "TMCO2_LND", "TMCO2_OCN", "TMQ", "TOT_CLD_VISTAU",
                "TREFHT", "TREFHTMN", "TREFHTMX", "TROP_P", "TROP_T", "TS",
                "TSMN", "TSMX", "U10", "U", "UQ", "UU", "V", "VD01", "VQ",
                "VT", "VU", "VV", "WGUSTD", "WSPDSRFMX", "WSUB", "Z3"]
monthly_train_vars = ["AODDUST1",
                "AODDUST2", "AODDUST3", "AODVIS", "AQRAIN", "AQSNOW", "AWNC", "AWNI", "bc_a1_SRF", "BURDENBC",
                "BURDENDUST", "BURDENPOM", "BURDENSEASALT", "BURDENSO4",
                "BURDENSOA", "CLDHGH", "CLDICE",
                "CLDLIQ", "CLDLOW", "CLDMED", "CLDTOT", "CLOUD", "DCQ",
                "dst_a1_SRF", "dst_a3_SRF", "DTCOND", "DTV", "EXTINCT",
                "FICE", "FLDS", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLUT",
                "FLUTC", "FREQI", "FREQL", "FREQR", "FREQS", "FSDS", "FSDSC",
                "FSNS", "FSNSC", "FSNT", "FSNTC", "FSNTOA", "FSNTOAC", "PBLH", "PHIS", "pom_a1_SRF", "PRECC",
                "PRECL", "PRECSC", "PRECSL", "Q", "QFLX", "QRL",
                "QRS", "RELHUM", "SFCO2", "SFCO2_FFF", "SFCO2_LND", "SFCO2_OCN", "SHFLX", "SNOWHICE", "SNOWHLND", "so4_a1_SRF", "so4_a2_SRF",
                "so4_a3_SRF", "soa_a1_SRF", "SWCF", "T", "TAUX", "TAUY", "TGCLDIWP", "TGCLDLWP", "TMCO2",
                "TMCO2_FFF", "TMCO2_LND", "TMCO2_OCN", "TMQ", "TREFHT", "TREFHTMN", "TREFHTMX", "TROP_P", "TROP_T", "TS",
                "TSMN", "TSMX", "V", "VD01", "VQ",
                "VT", "VU", "VV"]

monthly_test_vars = ["ABSORB", "ANRAIN", "AREI", "CCN3", "CO2", "CO2_FFF", "ICEFRAC", "ICIMR", "ICWMR", "IWC", "LANDFRAC", "OMEGA", "PS", "SOLIN", "SRFRAD", "TOT_CLD_VISTAU", "U10", "U", "WGUSTD", "WSPDSRFMX"]

monthly_validate_vars = ["ANSNOW", "AODABS", "AREL", "CDNUMC", "CO2_LND", "CO2_OCN", "ICLDIWP", "ICLDTWP", "LHFLX", "LWCF", "NUMICE", "NUMLIQ", "OCNFRAC", "OMEGAT", "PSL", "soa_a2_SRF", "UQ", "UU", "WSUB", "Z3"]

# train_vars, validate_vars = train_test_split(all_vars, test_size=0.2, random_state=3)
# monthly_train_vars, monthly_validate_vars = train_test_split(monthly_vars, test_size=0.2, random_state=3)
# train_vars, test_vars, = train_test_split(train_vars, test_size=0.25, random_state=3)  # 0.25 x 0.8 = 0.2
# monthly_train_vars, monthly_test_vars = train_test_split(monthly_train_vars, test_size=0.25, random_state=3)  # 0.25 x 0.8 = 0.2

def random_forest(X_train, X_test, y_train, y_test):
    params = {
        "max_depth": [2, 5],
        "random_state": [0]
    }
    # clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring="accuracy", cv=10)
    # clf.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100,
                                random_state=0, max_depth=5)
    rf.fit(X_train, y_train)

    # fn = data.feature_names
    # cn = data.target_names
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(rf.estimators_[0], ax=axes, filled=True);
    fig.savefig('rf_individualtree.png')
    y_pred = rf.predict(X_test)
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

    layers = [Dense(10, activation='relu', input_shape=(10,)),
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
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=params, scoring="accuracy", cv=10)

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

def count(element,seq):
    """Counts how often an element occurs
    ...in a sequence"""
    return sum(1 for i in seq if i == element)

def PredMostFrequent(X_train, X_test, y_train, y_test):
    (rf_preds, rf_acc) = random_forest(X_train, X_test, y_train, y_test)
    (nn_preds, nn_acc) = neural_net(X_train, X_test, y_train, y_test)
    (knn_preds, knn_acc, knn_params) = kNN(X_train, X_test, y_train, y_test)
    # (svm_preds, svm_acc) = SVM(X_train, X_test, y_train, y_test)
    (lda_preds, lda_acc) = LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    (qda_preds, qda_acc) = QuadraticDiscriminantAnalysis(X_train, X_test, y_train, y_test)
    y_pred = []
    for i in range(0,len(y_test)):
        # ERROR??? _counts not working???
        y_pred.append(statistics.mode([rf_preds[i],
                                       nn_preds[i],
                                       knn_preds[i],
                                       lda_preds[i],
                                       qda_preds[i]]))

    return(y_pred, sum(np.array(y_pred)==np.array(y_test)) / len(np.array(y_test)))


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dailyloc", help="location of optimal csv file",
                        type=str, default=f"../../data/minidata_calcs/minidata_daily_df.csv")
    parser.add_argument("-m", "--monthlyloc", help="location of monthly csv file",
                        type=str, default=None)
    parser.add_argument("-f", "--featureloc", help="location of feature list file",
                        type=str, default=None)
    parser.add_argument("-o", "--output", help="location of output csv file",
                        type=str, default=f"../../data/minidata_calcs/results.csv")
    parser.add_argument("-e", "--models", help="array of models to run",
                        nargs='+', required=True)
    parser.add_argument("-t", "--train", help="argument to train (if 0, use train set or variables, if 1, randomize)",
                        type=int, default=0)
    parser.add_argument("-r", "--reportdir", help="directory to place reports",
                        type=str, default=f'../../data/minidata_calcs/reports/')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parseArguments()
    argv_dailyloc = args.dailyloc
    argv_monthlyloc = args.monthlyloc
    argv_output = args.output
    argv_models = args.models
    argv_train = args.train
    argv_reportdir = args.reportdir
    argv_featureloc = args.featureloc

    count = 5
    if argv_dailyloc is not None:
        daily_df = pd.read_csv(argv_dailyloc)
    if argv_monthlyloc is not None:
        monthly_df = pd.read_csv(argv_monthlyloc)

    # just look at a particular algorithm and try and guess the level for now
    # remove some zfp_p_22 to balance the classes
    # x1, x2, y1, y2, idx1, idx2 = train_test_split(daily_df[daily_df["levels"] == 22], daily_df[daily_df["levels"]==22]["levels"], daily_df[daily_df["levels"] == 22].index, test_size=0.02, random_state=3)
    # daily_df.drop(index=idx1, inplace=True)
    # x1, x2, y1, y2, idx1, idx2 = train_test_split(monthly_df[monthly_df["levels"] == 22], monthly_df[monthly_df["levels"]==22]["levels"], monthly_df[monthly_df["levels"] == 22].index, test_size=0.02, random_state=3)
    # monthly_df.drop(index=idx1, inplace=True)

    if argv_dailyloc is not None:
        subset_daily = daily_df[daily_df["algs"] == "zfp"]
    if argv_monthlyloc is not None:
        subset_monthly = monthly_df[monthly_df["algs"] == "zfp"]
    if argv_dailyloc is not None:
        subset_daily = daily_df[daily_df["levels"] != 100000]
    if argv_monthlyloc is not None:
        subset_monthly = monthly_df[monthly_df["levels"] != 100000]
    #subset_daily = daily_df

    # load feature list from feature_list.pkl (set in feature_selection.py)
    if argv_featureloc is not None:
        with open(argv_featureloc, 'rb') as inp:
             features = pickle.load(inp)
    else:
        features = ["mean", "variance"]
    if argv_dailyloc is not None:
        # X1 = subset_daily[lcr_global_vars.features]
        X1 = subset_daily[features]

    if argv_monthlyloc is not None:
        # X2 = subset_monthly[lcr_global_vars.features]
        X2 = subset_monthly[features]

    if argv_dailyloc is not None:
        subset_daily["levels"][subset_daily["levels"] == 100000] = 28
    if argv_monthlyloc is not None:
        subset_monthly["levels"][subset_monthly["levels"] == 100000] = 28
    if argv_dailyloc is not None:
        y1_orig = subset_daily[["levels"]]
    if argv_monthlyloc is not None:
        y2_orig = subset_monthly[["levels"]]
    #y = subset_daily[["algs"]]
    #y = np.where(y == "zfp", 0, np.where(y == "sz", 1, 2))

    if argv_dailyloc is not None:
        y1 = np.array(y1_orig).ravel()
    if argv_monthlyloc is not None:
        y2 = np.array(y2_orig).ravel()

    if argv_dailyloc is not None:
        y1 = np.unique(y1, return_inverse=True)[1]
    if argv_monthlyloc is not None:
        y2 = np.unique(y2, return_inverse=True)[1]

    # create train-test split at random
    # X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                     y,
    #                                                     test_size = 0.33, random_state = 42)

    # create train-test split by selecting variables
    if argv_train == 0:
        if argv_dailyloc is not None:
            X1_train = X1[subset_daily["variable"].isin(train_vars)]
            X1_validate = X1[subset_daily["variable"].isin(validate_vars)]
            X1_test = X1[subset_daily["variable"].isin(test_vars)]

        if argv_monthlyloc is not None:
            X2_train = X2[subset_monthly["variable"].isin(monthly_train_vars)]
            X2_validate = X2[subset_monthly["variable"].isin(monthly_validate_vars)]
            X2_test = X2[subset_monthly["variable"].isin(monthly_test_vars)]

        if argv_dailyloc is not None:
            y1_train = y1[subset_daily["variable"].isin(train_vars)]
            y1_validate = y1[subset_daily["variable"].isin(validate_vars)]
            y1_test = y1[subset_daily["variable"].isin(test_vars)]

        if argv_monthlyloc is not None:
            y2_train = y2[subset_monthly["variable"].isin(monthly_train_vars)]
            y2_validate = y2[subset_monthly["variable"].isin(monthly_validate_vars)]
            y2_test = y2[subset_monthly["variable"].isin(monthly_test_vars)]
    elif argv_train == 1:

        if argv_dailyloc is not None:
            X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size = 0.2, random_state = 42)
            X1_train, X1_validate, y1_train, y1_validate = train_test_split(X1_train, y1_train, test_size=0.25, random_state=42)
        if argv_monthlyloc is not None:
            X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size = 0.2, random_state = 42)
            X2_train, X2_validate, y2_train, y2_validate = train_test_split(X2_train, y2_train, test_size=0.25, random_state=42)


    if argv_dailyloc is not None:
        X_train = X1_train
        X_validate = X1_validate
        X_test = X1_test

        y_train = y1_train
        y_validate = y1_validate
        y_test = y1_test
        if argv_monthlyloc is not None:
            X_train = X1_train.append(X2_train)
            X_validate = X1_validate.append(X2_validate)
            X_test = X1_test.append(X2_test)

            y_train = np.concatenate((y1_train, y2_train))
            y_validate = np.concatenate((y1_validate, y2_validate))
            y_test = np.concatenate((y1_test, y2_test))
    if argv_monthlyloc is not None:
        X_train = X2_train
        X_validate = X2_validate
        X_test = X2_test

        y_train = y2_train
        y_validate = y2_validate
        y_test = y2_test

    mappingto = list(set(y1_orig.T.values[0]))
    mappingfrom = list(set(y1))
    map = dict(zip(mappingfrom, mappingto))

    if "rf" in argv_models:
        (rf_preds, rf_acc) = random_forest(X_train, X_test, y_train, y_test)
        print("SECTION RANDOM FOREST -----------------")
        print(rf_acc)
        print(confusion_matrix(y_test, rf_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, rf_preds, output_dict=True)
        rf_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            rf_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        rf_df.rename(columnf_df = rf_df.res={"": "level"}, inplace=True)
        rf_df.to_csv(argv_reportdir + f'rf_report_{count}.csv', float_format="%.3f")
        print("END SECTION RANDOM FOREST -----------------")

    if "ada" in argv_models:
        (boost_preds, boost_acc) = adaboost(X_train, X_test, y_train, y_test)
        print("SECTION ADABOOST -----------------")
        print(boost_acc)
        print(confusion_matrix(y_test, boost_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, boost_preds, output_dict=True)
        boost_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            boost_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        boost_df.rename(columns={"": "level"}, inplace=True)
        boost_df.to_csv(argv_reportdir + f'boost_report_{count}.csv', float_format="%.3f")
        print("END SECTION ADABOOST -----------------")

    if "nn" in argv_models:
        (nn_preds, nn_acc) = neural_net(X_train, X_test, y_train, y_test)
        print("SECTION NEURAL NETWORK -----------------")
        print(nn_acc)
        print(confusion_matrix(y_test, nn_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, nn_preds, output_dict=True)
        nn_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            nn_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        nn_df.rename(columns={"": "level"}, inplace=True)
        nn_df.to_csv(argv_reportdir + f'nn_report_{count}.csv', float_format="%.3f")
        print("END SECTION NEURAL NETWORK -----------------")

    if "knn" in argv_models:
        print("SECTION KNN -----------------")
        (knn_preds, knn_acc, knn_params) = kNN(X_train, X_test, y_train, y_test)
        print(knn_acc)
        print(confusion_matrix(y_test, knn_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, knn_preds, output_dict=True)
        knn_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            knn_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        knn_df.rename(columns={"": "level"}, inplace=True)
        knn_df.to_csv(argv_reportdir + f'knn_report_{count}.csv', float_format="%.3f")
        print("END SECTION KNN -----------------")

    if "svm" in argv_models:
        print("SECTION SVM -----------------")
        (svm_preds, svm_acc) = SVM(X_train, X_test, y_train, y_test)
        print(svm_acc)
        print(confusion_matrix(y_test, svm_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, svm_preds, output_dict=True)
        svm_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            svm_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        svm_df.rename(columns={"": "level"}, inplace=True)
        svm_df.to_csv(argv_reportdir + f'svm_report_{count}.csv', float_format="%.3f")

        print("END SECTION SVM -----------------")

    if "lda" in argv_models:
        print("SECTION LDA -----------------")
        (lda_preds, lda_acc) = LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)
        print(lda_acc)
        print(confusion_matrix(y_test, lda_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, lda_preds, output_dict=True)
        lda_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            lda_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        lda_df.rename(columns={"": "level"}, inplace=True)
        lda_df.to_csv(argv_reportdir + f'lda_report_{count}.csv', float_format="%.3f")
        print("END SECTION LDA -----------------")

    if "qda" in argv_models:
        print("SECTION QDA -----------------")
        (qda_preds, qda_acc) = QuadraticDiscriminantAnalysis(X_train, X_test, y_train, y_test)
        print(qda_acc)
        print(confusion_matrix(y_test, qda_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, qda_preds, output_dict=True)
        qda_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            qda_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
 #       qda_df.rename(columns={"": "level"}, inplace=True)
        qda_df.to_csv(argv_reportdir + f'qda_report_{count}.csv', float_format="%.3f")

        print("END SECTION QDA -----------------")

    if "agg" in argv_models:
        print("SECTION AGGREGATE -----------------")
        (combine_preds, combine_acc) = PredMostFrequent(X_train, X_test, y_train, y_test)
        print(combine_acc)
        print(confusion_matrix(y_test, combine_preds))
        renamed_mat = pd.DataFrame(confusion_matrix(y_test, rf_preds))
        for i in range(len(renamed_mat)):
            renamed_mat.rename(index={i: f"{map[i]}"}, inplace=True)
            renamed_mat.rename(columns={i: f"{map[i]}"}, inplace=True)
        with open("../data_gathering/confusion_template.tex", "r") as template:
            lines = template.readlines()
        with open(argv_reportdir + "rf_template.tex", "w") as destination:
            for line in lines:
                for i in range(len(renamed_mat)):
                    nclasses = len(renamed_mat)
                    q = re.search(r"\\def\\numClasses\{9\}", line)
                    if q:
                        line = re.sub(r"9", str(nclasses), line)
                    classtring = r'\\def\\classNames\{\{"0","1","2","3","4","5","6","7","8"\}\}'
                    s = re.search(classtring, line)
                    if s:
                        # CHECK THIS
                        thing = {"','".join(str(x) for x in mappingto)}
                        line = re.sub('\{"0","1","2","3","4","5","6","7","8"\}', f'{thing}', line)
                        line = re.sub('\[', '{', line)
                        line = re.sub('\]', '}', line)
                        line = re.sub("'", '"', line)
                    string = f'%{{0,0,0,0,0,0,0,0,0,{i+1}}}'
                    p = re.search(string, line)
                    if p:
                        res = re.sub(string, f'{list(renamed_mat.iloc[i])}', line)
                        res = re.sub('\[', '{', res)
                        res = re.sub('\]', '}', res)
                        destination.write(res)
                    elif i==0:
                        destination.write(line)
        report = classification_report(y_test, combine_preds, output_dict=True)
        combine_df = pd.DataFrame(report).transpose()
        for i in range(len(map)):
            combine_df.rename(index={f"{i}": f"{map[i]}"}, inplace=True)
#        combine_df.rename(columns={"": "level"}, inplace=True)
        combine_df.to_csv(argv_reportdir + f'combine_report_{count}.csv', float_format="%.3f")
        print("END SECTION AGGREGATE -----------------")