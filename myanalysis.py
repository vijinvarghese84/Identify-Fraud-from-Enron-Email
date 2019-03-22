# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:56:57 2018

@author: Vijin
"""

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("final_project_dataset.pkl", "r") )
features = ["poi",'exercised_stock_options', 'total_stock_value']


#Removing TOTAL key
data_dict.pop('TOTAL',0)

data1 = featureFormat(data_dict, features)
label1, features_list1 = targetFeatureSplit(data1)

# To draw plot illustrating relationship between 2 features for POI and non-POI
def drawPlot(my_dataset1, features_analyze):
    data = featureFormat(my_dataset1, features_analyze)
    label, features_list = targetFeatureSplit(data)
    #print data

    poi_feature1 = [features_list[ii][0] for ii in range(0,len(features_list)) if label[ii] == 1]
    non_poi_feature1 = [features_list[ii][0] for ii in range(0,len(features_list)) if label[ii] == 0]
    poi_feature2 = [features_list[ii][1] for ii in range(0,len(features_list)) if label[ii] == 1]
    non_poi_feature2 = [features_list[ii][1] for ii in range(0,len(features_list)) if label[ii] == 0]
    
    plt.scatter(poi_feature1, poi_feature2, color='r')
    plt.scatter(non_poi_feature1, non_poi_feature2, color='b')

    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.show()
    
#Retrieve all features from dataset
def get_all_features(my_dataset1):
    feature_keys = my_dataset1['LAY KENNETH L'].keys()
    return feature_keys    
        