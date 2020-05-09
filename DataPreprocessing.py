import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from  copy import deepcopy
from collections import OrderedDict
from ast import literal_eval
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

class DataPreprocessingClass:
    def __init__(self, df, dpc = None):
        self.df = df   # initial data
        self.data = deepcopy(df) # clean data
        self.genres_dict = self.__generateGenresDict()
        if(type(dpc)==type(self)): # test case
            self.mostProfitableActors = dpc.mostProfitableActors
            self.dir_bins       = dpc.dir_bins
            self.yearMeanBudget = dpc.yearMeanBudget
            self.yearMeanRevenue = dpc.yearMeanRevenue
            self.directorProfit = dpc.directorProfit
            self.topKeywords    = dpc.topKeywords
            self.keywordProfit  = dpc.keywordProfit

        self.dataCleaning(dpc==None)
        # todo:
        # for test predefine:


    def __generateGenresDict(self):
        """
        Generates dictionary of all genres with it's id as key.
        :param df:
        :return:
        """
        res = {}
        all_genres = self.df['genres']
        for el in all_genres:
            LoD = strIntoLoD(el)
            for el in LoD:
                if (el['id'] not in res.keys()):
                    res[el['id']] = el['name']
        ordered_res = OrderedDict(sorted(res.items(), key=lambda item: item[0]))
        return ordered_res

    def __removeUnpopularIds(self, listOfIds, popularIds):
        """

        :param listOfIds: List of ids
        :param popularIds: Most popular agent over whole population
        :return: listOfIds without agents that aren't popular
        """
        res = []
        for id in listOfIds:
            if(id in popularIds.keys()):
                res.append(id)
        return res

    def __topNfromField(self, listOfIds, popularIds, N=3):
        """

        :param listOfIds: List of ids
        :param popularIds: Most popular agent over whole population
        :param N: Number of best (most popular, profitable, etc.) ids that we return
        :return:
        """
        idPopularity = {}
        for id in listOfIds:
            idPopularity[id] = popularIds[id]
        ordered_actorFilmNumber = OrderedDict(sorted(idPopularity.items(), key=lambda item: item[1], reverse=True))
        topN = [el[0] for el in list(ordered_actorFilmNumber.items())[:N]]
        return topN

    def __popularityFromField(self, field, threshold=3, topN = -1):
        res = {}
        for ids in self.data[field]:
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

    def __profitFromField(self, field, threshold_profit = 0, topN = -1):
        """
        Return dictionary of id:mean profit of id (topN of ids)
        :param field:
        :param threshold_profit:
        :return:
        """
        res = {}
        for index, row in self.data.iterrows():
            ids = row[field]
            for id in ids:
                if (id in res.keys()):
                    res[id]['profit'] += row['revenue']
                    res[id]['movies_num'] += 1
                else:
                    res[id] = {'profit': row['revenue'], 'movies_num': 1}
        topIds = {}

        for id in res.keys():
            if (res[id]['profit']/res[id]['movies_num'] >= threshold_profit):
                topIds[id] = res[id]['profit']/res[id]['movies_num']

        ordered_res = OrderedDict(sorted(topIds.items(), key=lambda item: item[1], reverse=True)[:topN])
        # print(f'Top keywords: {ordered_res}')
        return ordered_res

    def __generateAgentIdsDict(self, field):
        """

        :param field: string LoD
        :return: dictionary
        """
        res = {}
        field_data = self.data[field].apply(strIntoLoD).values
        for row in field_data:
            for el in row:
                res[el['id']]=el['name']
        return res

    def __getProfictCategory(self, value, bins):
        category = 0
        
        if(value<=bins[0]):
            return 0
        elif(value>bins[-1]):
            return len(bins)-1
        else:
            for i, bin in enumerate(bins):
                if(i!=0):
                    if(value > bins[i-1] and value <=  bins[i]):
                        return i-1


    def dataCleaning(self, train = True):

        # Revenue:
        self.data['revenue'] = pd.to_numeric(self.data['revenue'])
        self.data['logRevenue'] = self.data['revenue'].apply(math.log)


        if(train == True): # delete obviously not correct revenues
            to_deleted_rows = []
            for index, row in self.data.iterrows():
                if(row['revenue'] <= 3465): # box office of Dalida, from train set
                    to_deleted_rows.append(index)
            # Delete not unrepresentative rows
            for row in to_deleted_rows:
                self.data.drop(row, axis=0, inplace=True)


        columns = ['backdrop_path', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id',
                   'original_language', 'original_title', 'overview', 'popularity', 'poster_path',
                   'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime',
                   'spoken_languages', 'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count', 'Keywords',
                   'cast', 'crew']


        # Genres
        self.data['genresIDs'] = self.data['genres'].apply(strIntoLoD)
        self.data['genresIDs'] = self.data['genresIDs'].apply(getIDsFromListofDicts)

        # Production companies
        self.data['prodCompIDs'] = self.data['production_companies'].apply(strIntoLoD)
        self.data['prodCompIDs'] = self.data['prodCompIDs'].apply(getIDsFromListofDicts)

        # Collection
        self.data['collectionID'] = self.data['belongs_to_collection'].apply(dictOrNaN)
        self.data['collectionID'] = self.data['collectionID'].apply(lambda x: x['id'])
        self.data['isInCollection'] = self.data['belongs_to_collection'].apply(dictOrNaN)
        self.data['isInCollection'] = self.data['isInCollection'].apply(lambda x: int(x['id']!=-1))

        # Year
        self.data['year'] = self.data['release_date'].apply(lambda x: x.split('-')[0])
        self.data['year'] = pd.to_numeric(self.data["year"])



        # Month
        self.data['month'] = self.data['release_date'].apply(lambda x: x.split('-')[1])
        self.data['month'] = pd.to_numeric(self.data["month"])

        # Budget and revenue by year:
        if(train==True):
            self.yearMeanBudget = {}
            self.yearMeanRevenue = {}
            for index, row in self.data[['year','budget', 'revenue']].groupby('year').mean().iterrows():
                self.yearMeanBudget[index] = row['budget']
                self.yearMeanRevenue[index] = row['revenue']
        self.data['budget'] = self.data[['year','budget']].apply(lambda x: averageIfZero(x[1], x[0], self.yearMeanBudget), axis = 1 ,raw=True)




        # Most Popular Actors
        self.data['castIDs'] = self.data['cast'].apply(strIntoLoD)
        self.data['castIDs'] = self.data['castIDs'].apply(getIDsFromListofDicts)
        if(train == True):
            self.mostProfitableActors = self.__profitFromField(field='castIDs', threshold_profit=0, topN = 20000) # Actors with the most number of movies
            # print(f'top actors: {self.mostProfitableActors}')
            # print(len(self.mostProfitableActors))
        else:
            # in case of test, topActors should already be defined
            pass
        self.data['castIDs'] = self.data['castIDs'].apply(lambda x: self.__removeUnpopularIds(x, self.mostProfitableActors))
        self.data['castIDs'] = self.data['castIDs'].apply(lambda x: self.__topNfromField(x, self.mostProfitableActors, 40))
        self.data['topActorsNum'] = self.data['castIDs'].apply(len)
        # self.actorIds = self.__generateAgentIdsDict(field='cast')

        # Director
        self.data['crew'] = self.data['crew'].apply(strIntoLoD)
        self.data['director'] = self.data['crew'].apply(getDirector, args=('name', ))
        self.data['directorID'] = self.data['crew'].apply(getDirector, args=('id', ))
        # self.directorIds = dict([(row['directorID'], row['director']) for index, row in self.data.iterrows()])

        # Director categories:
        if(train):
            self.directorProfit = self.data.groupby('directorID').mean()['revenue'].to_dict()
            min_rev = min(self.directorProfit)
            max_rev = max(self.directorProfit)
            dir_bins_N = 10
            step = (max_rev - min_rev) / dir_bins_N
            self.dir_bins = np.arange(min_rev, max_rev, step)
            # dir_bins = np.arange(min_rev, max_rev-step, step)
            # dir_bins = np.append(dir_bins, np.arange(max_rev-step, max_rev, step/3))

            self.data['directorCat'] = self.data['directorID'].apply(
                lambda x: self.__getProfictCategory(self.directorProfit[x], self.dir_bins))
        else:
            # In this case self.dir_bins_N should be already defined, as well as self.directorProfit
            self.data['directorCat'] = self.data['directorID'].apply(
                lambda x: self.__getProfictCategory(self.directorProfit[x], self.dir_bins) if(x in self.directorProfit.keys()) else 5)


        # Most Popular keywords
        self.data['popularKeywordsIDs'] = self.data['Keywords'].apply(strIntoLoD)
        self.data['popularKeywordsIDs'] = self.data['popularKeywordsIDs'].apply(getIDsFromListofDicts)
        if(train == True):
            self.topKeywords = self.__popularityFromField(field='popularKeywordsIDs', threshold=3) # Keywords with the most number of movies
        else:
            # In this case self.topKeywords should be predefined
            pass
        self.data['popularKeywordsIDs'] = self.data['popularKeywordsIDs'].apply(lambda x: self.__removeUnpopularIds(x, self.topKeywords))
        self.data['popularKeywordsIDs'] = self.data['popularKeywordsIDs'].apply(lambda x: self.__topNfromField(x, self.topKeywords, 20))
        # self.keywordsIds = self.__generateAgentIdsDict(field='Keywords')


        # Most Profitable Keywords
        self.data['profitableKeywordsIDs'] = self.data['Keywords'].apply(strIntoLoD)
        self.data['profitableKeywordsIDs'] = self.data['profitableKeywordsIDs'].apply(getIDsFromListofDicts)
        if(train==True):
            self.keywordProfit = self.__profitFromField(field='profitableKeywordsIDs', threshold_profit=100000, topN=1000) # Keywords with biggest mean profits
        else:
            # In this case self.keywordProfit should be predefined
            pass
        self.data['profitableKeywordsIDs'] = self.data['profitableKeywordsIDs'].apply(lambda x: self.__removeUnpopularIds(x, self.keywordProfit))
        self.data['profitableKeywordsIDs'] = self.data['profitableKeywordsIDs'].apply(lambda x: self.__topNfromField(x, self.keywordProfit, 14))
        self.data['profitableKeywordsNum'] = self.data['profitableKeywordsIDs'].apply(len)



        # topNkeyIds = [el[0] for el in list(self.keywordProfit.items())[:50]]
        # topNWords = []
        # for wid in topNkeyIds:
        #     print([wid, self.keywordsIds[wid], self.keywordProfit[wid]])


        to_delete_cols = ['backdrop_path', 'homepage', 'poster_path', 'video',
                     'genres', 'production_companies', 'original_language',
                      'imdb_id', 'tagline', 'status',
                     'belongs_to_collection', 'release_date', 'original_title',
                     'crew', 'cast', 'Keywords',
                     # Can be used in the future vesions:
                     'spoken_languages', 'overview', 'production_countries',  ]

        # Delete low data:
        for col in to_delete_cols:
            self.data.drop(col, axis=1, inplace=True)

        # Show show_cols sorted by revenue
        # show_cols = ['revenue', 'title', 'year','directorCat','profitableKeywordsNum']
        # a = self.data.sort_values('revenue')[show_cols].values
        # for el in a:
        #     print(el)
        return self.data

def getDirector(LoD, nameOrId):
    for d in LoD:
        if(d['job']=='Director'):
            return d[nameOrId]
    if(nameOrId == 'name'): # In case that we don't know who director is
        return 'Vitaly Pankratov'
    else:
        return -1

def averageIfZero(budget, year, yearAverageBudgetDict):
    if(budget == 0):
        if(year in yearAverageBudgetDict.keys()):
            return yearAverageBudgetDict[year]
        else:
            closest_year_plus = year
            closest_year_minus = year
            while((closest_year_plus not in yearAverageBudgetDict.keys())
            and (closest_year_minus not in yearAverageBudgetDict.keys())):
                closest_year_plus += 1
                closest_year_minus -= 1
            if(closest_year_plus in yearAverageBudgetDict.keys()):
                return yearAverageBudgetDict[closest_year_plus]
            else:
                return yearAverageBudgetDict[closest_year_minus]
    else:
        return budget

def dictOrNaN(DoN):
    if(type(DoN)==type(8.5)):
        return {'id':-1, 'name':'No Collection'}
    else:
        return literal_eval(DoN)

def strIntoLoD(string):
    """
    :param string: string format: [{'id': 14, 'name': 'Fantasy'}, {'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]
    :return:
    """
    if(string[0] == '['):
        string = string[1:-1]
    LIST = string.split('}, ')
    res = []
    for i, el in enumerate(LIST):
        if(len(el)==0):
            continue
        if(i!=len(LIST)-1):
            el =  el + '}'
        res.append(literal_eval(el))
    return res

def getIDsFromListofDicts(LoD):
    """

    :param LoD: list of dictionaries [{'id': 14, 'name': 'Fantasy'}, {'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]
    :return: ids from this list
    """
    res = [el['id'] for el in LoD]
    return res

def main():
    df = pd.read_csv('data/train.tsv', sep='\t')
    dp = DataPreprocessingClass(df)
    # print(dp.data)
    # t = df.corr()
    # print(t)
    # ds = df[['budget', 'popularity', 'revenue', 'vote_average', 'vote_count']]
    #
    # pd.plotting.scatter_matrix(ds, figsize=(8, 8))
    # plt.show()
    #
    # plt.matshow(ds.corr())
    # plt.xticks(range(len(ds.columns)), ds.columns)
    # plt.yticks(range(len(ds.columns)), ds.columns)
    # plt.colorbar()
    # plt.show()
if __name__ == '__main__':
    main()