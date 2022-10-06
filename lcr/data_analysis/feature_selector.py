### Code from https://www.kaggle.com/code/mlwhiz/feature-selection-using-football-data/notebook
import pandas as pd
import numpy as np
import argparse
import sys
import pickle
sys.path.append('../data_gathering')
import lcr_global_vars as lcr_global_vars

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loc", help="location of optimal csv file",
                        type=str, default=None)
    parser.add_argument("-o", "--outfile", help="location of output feature list",
                        type=str, default=None)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parseArguments()
    argv_loc = args.loc
    argv_output = args.outfile

    num_feats=10
    if argv_loc is not None:
        df = pd.read_csv(argv_loc)
        subset_daily = df[df["algs"] == "zfp"]
        X = subset_daily[lcr_global_vars.features]
        y = subset_daily[["levels"]]
        y=y.squeeze()

    # Correlation Stuff
    #def cor_selector(X, y,num_feats=30):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]


    # Chi-2 Stuff
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import MinMaxScaler
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()

    # Recursive Feature Elimination

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()


    # Lasso
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression

    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=100), max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()


    # Tree-based
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()

    # LightGBM

    from sklearn.feature_selection import SelectFromModel
    # from lightgbm import LGBMClassifier
    #
    # lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
    #             reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    #
    # embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    # embeded_lgb_selector.fit(X, y)
    # embeded_lgb_support = embeded_lgb_selector.get_support()
    # embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    # print(str(len(embeded_lgb_feature)), 'selected features')

    # Feature List
    feature_name = list(X.columns)

    pd.set_option('display.max_rows', None)
    # put all selection together
    # feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
    #                                     'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                        'Random Forest':embeded_rf_support}) #,'LightGBM':embeded_lgb_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    sel = list(feature_selection_df["Total"] >= 4)
    with open(argv_output, 'wb+') as outp:
        flist = list(feature_selection_df[sel]["Feature"])
        print(flist)
        pickle.dump(flist, outp, pickle.HIGHEST_PROTOCOL)
