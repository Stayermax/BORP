import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from  copy import deepcopy
from collections import OrderedDict
from ast import literal_eval
import os
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"]=18,18

train_df = pd.read_csv('data/train.tsv', sep='\t')
test_df = pd.read_csv('data/train.tsv', sep='\t')


def __generateGenresDict(data):
    """
    Generates dictionary of all genres with it's id as key.
    :param df:
    :return:
    """
    res = {}
    all_genres = data['genres']
    for el in all_genres:
        LoD = strIntoLoD(el)
        for el in LoD:
            if (el['id'] not in res.keys()):
                res[el['id']] = el['name']
    ordered_res = OrderedDict(sorted(res.items(), key=lambda item: item[0]))
    return ordered_res


def __removeUnpopularIds(listOfIds, popularIds):
    """
    :param listOfIds: List of ids
    :param popularIds: Most popular agent over whole population
    :return: listOfIds without agents that aren't popular
    """
    res = []
    for id in listOfIds:
        if (id in popularIds.keys()):
            res.append(id)
    return res


def __topNfromField(listOfIds, popularIds, N=3):
    """

    :param listOfIds: List of ids
    :param popularIds: Most popular agent over whole population
    :param N: Number of best (most popular, profitable, etc.) ids that we return
    :return:
    """
    idPopularity = {}
    for id in listOfIds:
        idPopularity[id] = popularIds[id]
    ordered_agentFilmNumber = OrderedDict(sorted(idPopularity.items(), key=lambda item: item[1], reverse=True))
    topN = [el[0] for el in list(ordered_agentFilmNumber.items())[:N]]
    return topN


def __popularityFromField(data, field, threshold=3, topN=-1):
    res = {}
    for ids in data[field]:
        for id in ids:
            if (id in res.keys()):
                res[id] += 1
            else:
                res[id] = 1
    topIds = {}
    for id in res.keys():
        if (res[id] >= threshold):
            topIds[id] = res[id]
    ordered_res = OrderedDict(sorted(topIds.items(), key=lambda item: item[1], reverse=True))

    return ordered_res


def __profitFromField(data, field, threshold_profit=0, topN=-1):
    """
    Return dictionary of id:mean profit of id (topN of ids)
    :param field:
    :param threshold_profit:
    :return:
    """
    res = {}
    for index, row in data.iterrows():
        ids = row[field]
        for id in ids:
            if (id in res.keys()):
                res[id]['profit'] += row['revenue']
                res[id]['movies_num'] += 1
            else:
                res[id] = {'profit': row['revenue'], 'movies_num': 1}
    topIds = {}

    for id in res.keys():
        if (res[id]['profit'] / res[id]['movies_num'] >= threshold_profit):
            topIds[id] = res[id]['profit'] / res[id]['movies_num']

    ordered_res = OrderedDict(sorted(topIds.items(), key=lambda item: item[1], reverse=True)[:topN])
    # print(f'Top keywords: {ordered_res}')
    return ordered_res


def __generateAgentIdsDict(data, field):
    """

    :param field: string LoD
    :return: dictionary
    """
    res = {}
    field_data = data[field].apply(strIntoLoD).values
    for row in field_data:
        for el in row:
            res[el['id']] = el['name']
    return res


def __getProfictCategory(value, bins):
    category = 0

    if (value <= bins[0]):
        return 0
    elif (value > bins[-1]):
        return len(bins) - 1
    else:
        for i, bin in enumerate(bins):
            if (i != 0):
                if (value > bins[i - 1] and value <= bins[i]):
                    return i - 1


def getDirector(LoD, nameOrId):
    for d in LoD:
        if (d['job'] == 'Director'):
            return d[nameOrId]
    if (nameOrId == 'name'):  # In case that we don't know who director is
        return 'Vitaly Pankratov'
    else:
        return -1


def averageIfZero(budget, year, yearAverageBudgetDict):
    if (budget == 0):
        if (year in yearAverageBudgetDict.keys()):
            return yearAverageBudgetDict[year]
        else:
            closest_year_plus = year
            closest_year_minus = year
            while ((closest_year_plus not in yearAverageBudgetDict.keys())
                   and (closest_year_minus not in yearAverageBudgetDict.keys())):
                closest_year_plus += 1
                closest_year_minus -= 1
            if (closest_year_plus in yearAverageBudgetDict.keys()):
                return yearAverageBudgetDict[closest_year_plus]
            else:
                return yearAverageBudgetDict[closest_year_minus]
    else:
        return budget


def dictOrNaN(DoN):
    if (type(DoN) == type(8.5)):
        return {'id': -1, 'name': 'No Collection'}
    else:
        return literal_eval(DoN)


def strIntoLoD(string):
    """
    :param string: string format: [{'id': 14, 'name': 'Fantasy'}, {'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]
    :return:
    """
    if (string[0] == '['):
        string = string[1:-1]
    LIST = string.split('}, ')
    res = []
    for i, el in enumerate(LIST):
        if (len(el) == 0):
            continue
        if (i != len(LIST) - 1):
            el = el + '}'
        res.append(literal_eval(el))
    return res


def getIDsFromListofDicts(LoD):
    """

    :param LoD: list of dictionaries [{'id': 14, 'name': 'Fantasy'}, {'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]
    :return: ids from this list
    """
    res = [el['id'] for el in LoD]
    return res

def dataCleaning(data, train=True, pretrained_data={}):

    if (train != True):
        genres_dict = pretrained_data['genres_dict']
        mostProfitableActors = pretrained_data['mostProfitableActors']  # Ordered Dict actor id: actor mean proffit
        dir_bins = pretrained_data['dir_bins']  # List directors bins by profit
        yearMeanRevenue = pretrained_data['yearMeanRevenue']  # Dict year: year mean revenue
        directorProfit = pretrained_data['directorProfit']  # Dict director id : director mean profit
        topKeywords = pretrained_data['topKeywords']  # Ordered Dict keyword id: keyword film number
        keywordProfit = pretrained_data['keywordProfit']  # Ordered Dict keyword id: keyword mean profit
        mostProductiveCompanies = pretrained_data[
            'mostProductiveCompanies']  # Ordered Dict Company Id: Number of company films
        companiesIds = pretrained_data['companiesIds']  # Dict Company Id : Company Name
    else:
        genres_dict = __generateGenresDict(data)

    # Genres
    data['genresIDs'] = data['genres'].apply(strIntoLoD)
    data['genresIDs'] = data['genresIDs'].apply(getIDsFromListofDicts)

    # Production companies
    data['prodCompIDs'] = data['production_companies'].apply(strIntoLoD)
    data['prodCompIDs'] = data['prodCompIDs'].apply(getIDsFromListofDicts)

    # Collection
    data['isInCollection'] = data['belongs_to_collection'].apply(lambda x: int(type(x) != type(0.1)))

    # Year
    data['year'] = data['release_date'].apply(lambda x: x.split('-')[0])
    data['year'] = pd.to_numeric(data["year"])

    # Month
    data['month'] = data['release_date'].apply(lambda x: x.split('-')[1])
    data['month'] = pd.to_numeric(data["month"])

    # revenue by year:
    if (train == True):
        yearMeanRevenue = {}
        for index, row in data[['year', 'revenue']].groupby('year').mean().iterrows():
            yearMeanRevenue[index] = row['revenue']

            # Companies: We took top 10 companies with the biggest number of films
    data['companiesIDs'] = data['production_companies'].apply(strIntoLoD)
    data['companiesIDs'] = data['companiesIDs'].apply(getIDsFromListofDicts)
    if (train == True):
        mostProductiveCompanies = __popularityFromField(data, 'companiesIDs', threshold=0, topN=5)
        companiesIds = __generateAgentIdsDict(data, field='production_companies')
    else:
        # in case of test, mostProductiveCompanies should already be defined
        pass
    data['companiesIDs'] = data['companiesIDs'].apply(lambda x: __removeUnpopularIds(x, mostProductiveCompanies))

    # Most Popular Actors
    data['castIDs'] = data['cast'].apply(strIntoLoD)
    data['castIDs'] = data['castIDs'].apply(getIDsFromListofDicts)
    if (train == True):
        mostProfitableActors = __profitFromField(data, field='castIDs', threshold_profit=0,
                                                 topN=20000)  # Actors with the most number of movies
    else:
        # in case of test, topActors should already be defined
        pass
    data['castIDs'] = data['castIDs'].apply(lambda x: __removeUnpopularIds(x, mostProfitableActors))
    data['castIDs'] = data['castIDs'].apply(lambda x: __topNfromField(x, mostProfitableActors, 40))
    data['topActorsNum'] = data['castIDs'].apply(len)
    # actorIds = __generateAgentIdsDict(field='cast')

    # Director
    data['crew'] = data['crew'].apply(strIntoLoD)
    data['director'] = data['crew'].apply(getDirector, args=('name',))
    data['directorID'] = data['crew'].apply(getDirector, args=('id',))
    # directorIds = dict([(row['directorID'], row['director']) for index, row in data.iterrows()])

    # Director categories:
    if (train):
        directorProfit = data.groupby('directorID').mean()['revenue'].to_dict()
        min_rev = min(directorProfit)
        max_rev = max(directorProfit)
        dir_bins_N = 10
        step = (max_rev - min_rev) / dir_bins_N
        dir_bins = np.arange(min_rev, max_rev, step)

        data['directorCat'] = data['directorID'].apply(
            lambda x: __getProfictCategory(directorProfit[x], dir_bins))
    else:
        # In this case dir_bins_N should be already defined, as well as directorProfit
        data['directorCat'] = data['directorID'].apply(
            lambda x: __getProfictCategory(directorProfit[x], dir_bins) if (x in directorProfit.keys()) else 5)

    # Most Popular keywords
    data['popularKeywordsIDs'] = data['Keywords'].apply(strIntoLoD)
    data['popularKeywordsIDs'] = data['popularKeywordsIDs'].apply(getIDsFromListofDicts)
    if (train == True):
        topKeywords = __popularityFromField(data, field='popularKeywordsIDs',
                                            threshold=3)  # Keywords with the most number of movies
    else:
        # In this case topKeywords should be predefined
        pass
    data['popularKeywordsIDs'] = data['popularKeywordsIDs'].apply(lambda x: __removeUnpopularIds(x, topKeywords))
    data['popularKeywordsIDs'] = data['popularKeywordsIDs'].apply(lambda x: __topNfromField(x, topKeywords, 20))
    # keywordsIds = __generateAgentIdsDict(field='Keywords')

    # Most Profitable Keywords
    data['profitableKeywordsIDs'] = data['Keywords'].apply(strIntoLoD)
    data['profitableKeywordsIDs'] = data['profitableKeywordsIDs'].apply(getIDsFromListofDicts)
    if (train == True):
        keywordProfit = __profitFromField(data, field='profitableKeywordsIDs', threshold_profit=100000,
                                          topN=1000)  # Keywords with biggest mean profits
    else:
        # In this case keywordProfit should be predefined
        pass
    data['profitableKeywordsIDs'] = data['profitableKeywordsIDs'].apply(
        lambda x: __removeUnpopularIds(x, keywordProfit))
    data['profitableKeywordsIDs'] = data['profitableKeywordsIDs'].apply(
        lambda x: __topNfromField(x, keywordProfit, 14))
    data['profitableKeywordsNum'] = data['profitableKeywordsIDs'].apply(len)

    # topNkeyIds = [el[0] for el in list(keywordProfit.items())[:50]]
    # topNWords = []
    # for wid in topNkeyIds:
    #     print([wid, keywordsIds[wid], keywordProfit[wid]])

    to_delete_cols = ['backdrop_path', 'homepage', 'poster_path', 'video',
                      'genres', 'production_companies', 'original_language',
                      'imdb_id', 'tagline', 'status',
                      'belongs_to_collection', 'release_date', 'original_title',
                      'crew', 'cast', 'Keywords', 'production_countries',
                      # Can be used in the future vesions:
                      'spoken_languages', 'overview', ]

    # Delete low data:
    for col in to_delete_cols:
        data.drop(col, axis=1, inplace=True)

    # # show_cols sorted by revenue
    # show_cols = ['revenue', 'title', 'year','directorCat','profitableKeywordsNum']
    # a = data.sort_values('revenue')[show_cols].values
    # for el in a:
    #     print(el)
    trained_data = {}
    if (train == True):
        trained_data['genres_dict'] = genres_dict
        trained_data['mostProfitableActors'] = mostProfitableActors
        trained_data['dir_bins'] = dir_bins
        trained_data['yearMeanRevenue'] = yearMeanRevenue
        trained_data['directorProfit'] = directorProfit
        trained_data['topKeywords'] = topKeywords
        trained_data['keywordProfit'] = keywordProfit
        trained_data['mostProductiveCompanies'] = mostProductiveCompanies
        trained_data['companiesIds'] = companiesIds

    return data, trained_data

clear_train_df, trained_data  = dataCleaning(data=train_df, train=True)
clear_test_df, zero_data  = dataCleaning(data=test_df, train=False, pretrained_data=trained_data)

def generateFeatures(trained_data, df, cont_features_list, category_list, Train = False):
    X = []
    Features_names = deepcopy(cont_features_list)
    for index, row in df.iterrows():
        Features_names = deepcopy(cont_features_list)
        X.append(list(row[cont_features_list].values))
        # add year avrg revenue
        cur_year = row['year']
        while(cur_year not in trained_data['yearMeanRevenue'].keys()):
            cur_year += 1
        Features_names.append('yearMeanRevenue')
        X[-1].append(trained_data['yearMeanRevenue'][cur_year])
        if('month' in category_list):
            for i in range(12):
                Features_names.append(f'month_{i+1}')
                if(row['month']==i+1):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if ('directorCat' in category_list):
            for i in range(len(trained_data['dir_bins'])):
                Features_names.append(f'dirCat_{i}')
                if (row['directorCat'] == i):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if('genresIDs' in category_list):
            # print(f'row genres: {row["genresIDs"]}')
            # print(f'genres_dict: {train_dp.genres_dict.items()}')
            for i, id_genre in enumerate(trained_data['genres_dict'].items()):
                Features_names.append(f'genre_{id_genre[0]}')
                if (id_genre[0] in row['genresIDs']):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
        if('companiesIDs' in category_list):
            for cid in trained_data['mostProductiveCompanies'].keys():
                Features_names.append(f'company_{cid}')
                if (cid in row['companiesIDs']):
                    X[-1].append(1)
                else:
                    X[-1].append(0)
    return X


num_feats = ['popularity', 'vote_count', 'isInCollection',
             'profitableKeywordsNum', 'topActorsNum', 'year', ]
cat_feats = ['directorCat', 'month', 'genresIDs', 'companiesIDs']

print('Features creation started')
# clear_train_df
# clear_test_df
import pickle
train_X_feat_path = 'no_classes_save/train_X.pkl'
if (os.path.exists(train_X_feat_path)):
    train_X = pickle.load(open(train_X_feat_path, "rb"))
else:
    train_X = generateFeatures(trained_data, clear_train_df, num_feats, cat_feats)
    pickle.dump(train_X, open(train_X_feat_path, "wb"))
train_y = list(clear_train_df['revenue'].values)

test_X_feat_path = 'no_classes_save/test_X.pkl'
if (os.path.exists(test_X_feat_path)):
    test_X = pickle.load(open(test_X_feat_path, "rb"))
else:
    test_X = generateFeatures(trained_data, clear_test_df, num_feats, cat_feats)
    pickle.dump(test_X, open(test_X_feat_path, "wb"))
test_y = list(clear_test_df['revenue'].values)

