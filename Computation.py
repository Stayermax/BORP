from DataPreprocessing import DataPreprocessingClass
import pandas as pd
import numpy as np
import pickle
import os
import time
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor

# Model list:
# 1) Base model - returns movie's year average revenue
# RMSLE is: 3.083787

# 2) First Model - linearRegression ( a lot of values < 0)
# RMSLE is: 2.654178

# 3) Second Model
    # budget 6.86
    # + popularity 6.88
    # + vote_average 7.07
    # - vote_average + vote_count 6.984
    # + profitableKeywordsNum 6.919
    # + topActorsNum year 7.02
    # + 'month', 'genresIDs'

# 4) Second model;RandomForestRegressor  2.148965
#     + all features 2.058
#     + depth 15 - random_state = 0 1.14
#     + depth 25 , n_estimators = 200

def generateFeatures(train_dp:DataPreprocessingClass, df, cont_features_list, category_list, Train = False):
    X = []
    Features_names = deepcopy(cont_features_list)
    for index, row in df.iterrows():
        Features_names = deepcopy(cont_features_list)
        X.append(list(row[cont_features_list].values))
        # add year avrg revenue
        cur_year = row['year']
        while(cur_year not in train_dp.yearMeanRevenue.keys()):
            cur_year += 1
        Features_names.append('yearMeanRevenue')
        X[-1].append(train_dp.yearMeanRevenue[cur_year])
        if('month' in category_list):
            for i in range(12):
                Features_names.append(f'month_{i+1}')
                if(row['month']==i+1):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if ('directorCat' in category_list):
            for i in range(len(train_dp.dir_bins)):
                Features_names.append(f'dirCat_{i}')
                if (row['directorCat'] == i):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if('genresIDs' in category_list):
            # print(f'row genres: {row["genresIDs"]}')
            # print(f'genres_dict: {train_dp.genres_dict.items()}')
            for i, id_genre in enumerate(train_dp.genres_dict.items()):
                Features_names.append(f'genre_{id_genre[0]}')
                if (id_genre[0] in row['genresIDs']):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if('companiesIDs' in category_list):
            for cid in train_dp.mostProductiveCompanies.keys():
                Features_names.append(f'company_{cid}')
                if (cid in row['companiesIDs']):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
    return X

### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

def predict(train_df, test_df):
    train_df = pd.read_csv('data/train.tsv', sep='\t')
    if (os.path.exists('pickle_saves/train_dp.pkl')):
        train_dp = pickle.load(open('pickle_saves/train_dp.pkl', "rb"))
        print('Train data loaded')
    else:
        train_dp = DataPreprocessingClass(train_df)
        pickle.dump(train_dp, open('pickle_saves/train_dp.pkl', "wb"))
        print('Train data evaluated')

    test_df = pd.read_csv('data/train.tsv', sep='\t')
    if (os.path.exists('pickle_saves/test_dp.pkl')):
        test_dp = pickle.load(open('pickle_saves/test_dp.pkl', "rb"))
        print('Test data loaded')
    else:
        test_dp = DataPreprocessingClass(test_df, train_dp)
        pickle.dump(train_dp, open('pickle_saves/test_dp.pkl', "wb"))
        print('Test data evaluated')

    num_feats = ['budget', 'popularity', 'vote_average', 'vote_count', 'isInCollection',
                 'profitableKeywordsNum', 'topActorsNum', 'year', 'month']
    cat_feats = ['directorCat', 'month', 'genresIDs', 'companiesIDs']

    print('Features creation started')
    clear_train_df = train_dp.data
    clear_test_df = test_dp.data

    train_X_feat_path = 'pickle_saves/train_X_comp.pkl'
    if (os.path.exists(train_X_feat_path)):
        train_X = pickle.load(open(train_X_feat_path, "rb"))
    else:
        train_X = generateFeatures(train_dp, clear_train_df, num_feats, cat_feats)
        pickle.dump(train_X, open(train_X_feat_path, "wb"))
    train_y = list(train_dp.data['revenue'].values)

    test_X_feat_path = 'pickle_saves/test_X_comp.pkl'
    if (os.path.exists(test_X_feat_path)):
        test_X = pickle.load(open(test_X_feat_path, "rb"))
    else:
        test_X = generateFeatures(train_dp, clear_test_df, num_feats, cat_feats)
        pickle.dump(test_X, open(test_X_feat_path, "wb"))
    test_y = list(clear_test_df['revenue'].values)

    print('Features creation finished')

    model = RandomForestRegressor(criterion='mae')
    model_path = 'pickle_saves/RFR_comp.pkl'
    if (os.path.exists(model_path)):
        print('RFR model loaded')
        model = pickle.load(open(model_path, "rb"))
    else:
        model.fit(train_X, train_y)
        pickle.dump(model, open(model_path, "wb"))

    pred_dict = {'id': [], 'rev': [], 'pred_rev': []}
    pred_dict['id'] = list(clear_test_df['id'])
    pred_dict['rev'] = list(clear_test_df['revenue'])
    pred_dict['pred_rev'] = model.predict(test_X)

    prediction_df = pd.DataFrame.from_dict(pred_dict)
    # ### Example - Calculating RMSLE
    res = rmsle(prediction_df['rev'], prediction_df['pred_rev'])
    print("RMSLE is: {:.6f}".format(res))

def main():
    train_df = pd.read_csv('data/train.tsv', sep='\t')
    dp_model_path = 'RandomForest models/dp_model.pkl'
    if(os.path.exists(dp_model_path)):
        train_dp = pickle.load( open(dp_model_path, "rb" ))
        print('Train data loaded')
    else:
        train_dp = DataPreprocessingClass(train_df)
        pickle.dump(train_dp, open(dp_model_path, "wb"))
        print('Train data evaluated')

    test_df = pd.read_csv('data/train.tsv', sep='\t')
    test_dp = DataPreprocessingClass(test_df, train_dp)
    #
    # if (os.path.exists('pickle_saves/test_dp.pkl')):
    #     test_dp = pickle.load(open('pickle_saves/test_dp.pkl', "rb"))
    #     print('Test data loaded')
    # else:
    #     test_dp = DataPreprocessingClass(test_df, train_dp)
    #     pickle.dump(train_dp, open('pickle_saves/test_dp.pkl', "wb"))
    #     print('Test data evaluated')
    #

    num_feats = ['budget', 'popularity', 'vote_average', 'vote_count', 'isInCollection',
                               'profitableKeywordsNum', 'topActorsNum', 'year', 'month']
    cat_feats = ['directorCat', 'month', 'genresIDs', 'companiesIDs']

    print('Features creation started')

    clear_train_df = train_dp.data
    clear_test_df  = test_dp.data

    train_X_feat_path = 'pickle_saves/train_X_comp.pkl'
    if (os.path.exists(train_X_feat_path)):
        train_X = pickle.load(open(train_X_feat_path, "rb"))
    else:
        train_X = generateFeatures(train_dp, clear_train_df, num_feats, cat_feats)
        pickle.dump(train_X, open(train_X_feat_path, "wb"))
    train_y = list(train_dp.data['revenue'].values)

    test_X_feat_path = 'pickle_saves/test_X_comp.pkl'
    test_X = generateFeatures(train_dp, clear_test_df, num_feats, cat_feats)
    #
    # if (os.path.exists(test_X_feat_path)):
    #     test_X = pickle.load(open(test_X_feat_path, "rb"))
    # else:
    #     test_X = generateFeatures(train_dp, clear_test_df, num_feats, cat_feats)
    #     pickle.dump(test_X, open(test_X_feat_path, "wb"))
    test_y = list(clear_test_df['revenue'].values)

    print('Features creation finished')

    model = RandomForestRegressor(criterion='mae')
    model_path = 'pickle_saves/RFR_comp.pkl'
    if (os.path.exists(model_path)):
        print('RFR model loaded')
        model = pickle.load(open(model_path, "rb"))
    else:
        model.fit(train_X, train_y)
        pickle.dump(model, open(model_path, "wb"))

    pred_dict = {'id': [], 'rev': [], 'pred_rev': []}
    pred_dict['id'] = list(clear_test_df['id'])
    pred_dict['rev'] = list(clear_test_df['revenue'])
    pred_dict['pred_rev'] = model.predict(test_X)

    prediction_df = pd.DataFrame.from_dict(pred_dict)
    res = rmsle(prediction_df['rev'], prediction_df['pred_rev'])
    print("RMSLE is: {:.6f}".format(res))


if __name__ == '__main__':
    main()