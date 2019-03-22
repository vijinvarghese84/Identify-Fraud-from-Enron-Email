# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:24:43 2018

@author: Vijin
"""
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
from sklearn.feature_selection import SelectKBest,SelectPercentile

#Retrieving k best features using SelectKBest
def get_k_best(dictionary, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection returning:
    {feature:score}
    """
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    pairs = zip(features_list[1:], scores)
    #combined scores and features into a pandas dataframe then sort 
    k_best_features = pd.DataFrame(pairs,columns = ['feature','score'])
    k_best_features = k_best_features.sort_values('score',ascending = False)
    
    return k_best_features[:k]

 