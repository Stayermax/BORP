from DataPreprocessing import DataPreprocessingClass
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import SVR


# Model list:
# 1) Base model - returns movie's year average revenue
# RMSLE is: 3.083787

# 2) First Model - linearRegression ( a lot of values < 0)
# RMSLE is: 2.654178

# 3) Second Model

def generateFeatures(train_dp:DataPreprocessingClass, df, cont_features_list, category_list):
    X = []
    for index, row in df.iterrows():
        X.append(list(row[cont_features_list].values))
        # add year avrg revenue
        cur_year = row['year']
        while(cur_year not in train_dp.yearMeanRevenue.keys()):
            cur_year += 1
        X[-1].append(train_dp.yearMeanRevenue[cur_year])
        if('month' in category_list):
            for i in range(12):
                if(row['month']==i+1):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if ('directorCat' in category_list):
            for i in range(len(train_dp.dir_bins)-1):
                if (row['directorCat'] == i):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if('genresIDs' in category_list):
            # print(f'row genres: {row["genresIDs"]}')
            # print(f'genres_dict: {train_dp.genres_dict.items()}')
            for i, id_genre in enumerate(train_dp.genres_dict.items()):
                if (id_genre[0] in row['genresIDs']):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
    return X

class Model():
    def __init__(self):
        self.Trained = False
        pass

    def train(self, train_dp):
        self.Trained = True
        pass

    def predict(self, row):
        pass

class BaseModel(Model):
    def train(self, train_dp):
        self.Trained = True
        self.train_dp = train_dp

    def predict(self, row):
        return self.train_dp.yearMeanRevenue[row['year']]

class Model1(Model):
    def train(self, train_dp, load=True):
        self.Trained = True
        self.cont_features_list = ['budget', 'popularity', 'vote_average', 'vote_count', 'isInCollection',
                               'profitableKeywordsNum', 'topActorsNum', 'year', 'month']
        self.category_list = ['directorCat', 'month', 'genresIDs']
        self.train_dp = train_dp
        model_path = 'pickle_saves/model_1.p'
        if (os.path.exists() and load):
            print('Model 1 loaded')
            self.model = pickle.load( open(model_path, "rb" ))
        else:
            X = generateFeatures(train_dp, train_dp.data, self.cont_features_list, self.category_list)
            y = list(train_dp.data['revenue'].values)

            self.model = LinearRegression().fit(X,y)
            pickle.dump(self.model, open(model_path, "wb"))
            print('Model 1 trained')
        print(f'Model coefs: {self.model.coef_}')


    def predict(self, row):
        vals = generateFeatures(self.train_dp, row, self.cont_features_list, self.category_list)
        return self.model.predict(vals)[0]

class Model2(Model):
    def train(self, train_dp, load=True):
        self.Trained = True
        self.cont_features_list = ['budget', 'popularity', 'vote_average', 'vote_count', 'isInCollection',
                               'profitableKeywordsNum', 'topActorsNum', 'year', 'month']
        self.category_list = ['directorCat', 'month', 'genresIDs']
        self.train_dp = train_dp
        model_path = 'pickle_saves/model_2.p'
        if (os.path.exists(model_path) and load):
            print('Model 2 loaded')
            self.model = pickle.load( open( model_path, "rb" ))
        else:
            X = generateFeatures(train_dp, train_dp.data, self.cont_features_list, self.category_list)
            # print(X)
            y = list(train_dp.data['revenue'].values)
            self.model = SVR(degree=1, C=1.0, epsilon=0.6).fit(X,y)
            # self.model = RidgeCV(alphas=np.logspace(0.5, 1, 25)).fit(X,y)

            pickle.dump(self.model, open(model_path, "wb"))
            print('Model 2 trained')
        # print(f'Model coefs: {self.model.coef_}')

    def predict(self, row):
        vals = generateFeatures(self.train_dp, row, self.cont_features_list, self.category_list)
        # print(vals)
        return self.model.predict(vals)[0]


def predict(model, train_dp, row_df):
    if(model.Trained == False):
        model.train(train_dp)
    prediction = model.predict(row_df)
    if(prediction<0):
        # print(row_df)
        return train_dp.yearMeanRevenue[row_df['year'].values[0]]
    else:
        return prediction

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

def main(modelName = 'Second Model'):
    train_df = pd.read_csv('data/train.tsv', sep='\t')
    if(os.path.exists('pickle_saves/train_dp.p')):
        train_dp = pickle.load( open( 'pickle_saves/train_dp.p', "rb" ))
    else:
        train_dp = DataPreprocessingClass(train_df)
        pickle.dump(train_dp, open('pickle_saves/train_dp.p', "wb"))

    test_df = pd.read_csv('data/train.tsv', sep='\t')
    if (os.path.exists('pickle_saves/test_dp.p')):
        test_dp = pickle.load(open('pickle_saves/test_dp.p', "rb"))
    else:
        test_dp = DataPreprocessingClass(test_df, train_dp)
        pickle.dump(train_dp, open('pickle_saves/test_dp.p', "wb"))

    if (modelName == 'Base Model'):
        model = BaseModel()
    elif (modelName == 'First Model'):
        model = Model1()
    elif (modelName == 'Second Model'):
        model = Model2()

    prediction_df = pd.DataFrame(columns=['id', 'revenue'])
    prediction_df['id'] = test_dp.data['id']

    pred_dict = {'id': [], 'rev':[], 'pred_rev':[]}
    for index, row in test_dp.data.iterrows():
        pred_dict['id'].append(row['id'])
        pred_dict['rev'].append(row['revenue'])
        pred_dict['pred_rev'].append(predict(model,train_dp, test_dp.data.loc[[index]]))
        # if (pred_dict['pred_rev'][-1]<0):
        #     print(pred_dict['pred_rev'][-1])
    prediction_df = pd.DataFrame.from_dict(pred_dict)
    # pd.set_option('display.max_rows', 1000)
    # print(prediction_df.head(1000))

    # ### Example - Calculating RMSLE
    res = rmsle(prediction_df['rev'], prediction_df['pred_rev'])
    print("RMSLE is: {:.6f}".format(res))
    print(type(model))

if __name__ == '__main__':
    main()