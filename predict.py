import argparse
import numpy as np
import pandas as pd
import os
import pickle

from Straight_code import generateFeatures, dataCleaning

### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """8l  =ll4444444444444444444444444
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))
# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

test_df = pd.read_csv(args.tsv_path, sep='\t')
print(f'predict.py')

######################################
### P R E D I C T I O N    P A R T ###
######################################


trained_data =  pickle.load(open('Trained_data/pretrained_data.pkl', "rb"))
clear_test_df, zero_data  = dataCleaning(data=test_df, train=False, pretrained_data=trained_data)

num_feats = ['popularity', 'vote_count', 'isInCollection',
             'profitableKeywordsNum', 'topActorsNum', 'year' ]
cat_feats = ['directorCat', 'month', 'genresIDs', 'companiesIDs']

print('Test features creation started')
test_X = generateFeatures(trained_data, clear_test_df, num_feats, cat_feats)

print(""" CatBoostRegressor START """)
from catboost import CatBoostRegressor

model_path = 'catboost models/best_model.pkl'
print('Catboost model loaded')
model = pickle.load(open(model_path, "rb"))

######################################
### S C O R I N G          P A R T ###
######################################
# prediction_df = pd.DataFrame(columns=['id', 'revenue'])
# prediction_df['id'] = test_df['id']
# prediction_df['revenue'] = test_df['revenue'].mean()

pred_dict = {'id': [], 'real_revenue': [], 'revenue': []}
pred_dict['id'] = list(clear_test_df['id'])
pred_dict['real_revenue'] = list(clear_test_df['revenue'])
pred_dict['revenue'] = model.predict(test_X)

prediction_df = pd.DataFrame.from_dict(pred_dict)
res = rmsle(prediction_df['real_revenue'], prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(res))

# Export prediction results
prediction_df[['id', 'revenue']].to_csv("prediction.csv", index=False, header=False)

print('THE END ')
