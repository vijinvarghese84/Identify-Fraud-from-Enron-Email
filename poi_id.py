    #!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from newfeatures import fraction_from_poi_feature, fraction_to_poi_feature
from myanalysis import drawPlot, get_all_features
from select_best_features import get_k_best
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers from main data
    data_dict.pop('TOTAL',0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Adding new features fraction_from_poi and fraction_to_poi to my_dataset
my_dataset = fraction_from_poi_feature(my_dataset)
my_dataset = fraction_to_poi_feature(my_dataset)

#Retrieving all features (including poi)
all_features = get_all_features(my_dataset)

#Removing email_address and poi since email address doesn't add any value and poi is a label
all_features.remove('email_address')
all_features.remove('poi')

# Adding all required features to features_list
features_list = features_list+all_features

print 'All features in feature list are: ', features_list

#Analyzing the newly added features
'''
features_analyze=['poi','fraction_from_poi','fraction_to_poi']
drawPlot(my_dataset, features_analyze)        
'''
      
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


#Selecting 15 best features using SelectKBest
k_best_features = SelectKBest(k=15)

#Finding the best features using SelectKBest and their scores
'''
best_features = get_k_best(my_dataset,features_list,15)
print "Best Features are ", best_features
'''
#Dimensionality reduction using PCA
'''
pca = PCA(n_components=0.9)
'''
# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data

#Using Pipeline, first transforming using MinMax scaler, then using Select KBest and 
#finally fitting using DecisionTree classifier with parameters tuned
decisiontree_kbest = Pipeline([('scaler', MinMaxScaler()),
        ('kbest', k_best_features),
                 ('tree', DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=1))])

#Using Pipeline, first transforming using MinMax scaler, then using Select KBest and 
#finally fitting using RandomForest classifier with parameters tuned  
randomforest_kbest = Pipeline([('scaler', MinMaxScaler()),
        ('kbest', k_best_features),
                 ('randomforest', RandomForestClassifier(max_features=0.8 , n_estimators=5,min_samples_leaf=1))])  
#Using Pipeline, first transforming using MinMax scaler, then using Select KBest and 
#finally fitting using SVM classifier with parameters tuned     
svm_kbest = Pipeline([('scaler', MinMaxScaler()),
        ('kbest', k_best_features),
                 ('svm', svm.SVC(C=1, gamma=10, kernel='poly'))])   
    
#Using Pipeline, first transforming using MinMax scaler, then using PCA and 
#finally fitting using DecisionTree classifier with parameters tuned   
decisiontree_pca = Pipeline([('pca', pca),
                 ('tree', DecisionTreeClassifier(max_depth=100, min_samples_split=10, min_samples_leaf=1))])

#Using Pipeline, first transforming using MinMax scaler, then using PCA and 
#finally fitting using RandomForest classifier with parameters tuned    
randomforest_pca = Pipeline([('pca', pca),
                 ('randomforest', RandomForestClassifier(max_features=0.75, n_estimators=5,min_samples_leaf=1))])  
#Using Pipeline, first transforming using MinMax scaler, then using PCA and 
#finally fitting using SVM classifier with parameters tuned    
svm_pca = Pipeline([('scaler', MinMaxScaler()),
        ('pca', pca),
                 ('svm', svm.SVC(C=1, gamma=10, kernel='poly'))])   
    

 
    

#decisiontree_pipe.fit(features, labels)

#pipe.predict(newdata)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

###Parameters for tuning
'''    
decision_parameters = {'tree__max_depth':[10,20,50,100],'tree__min_samples_split':[2,4,5,8,10],'tree__min_samples_leaf':[1,2,4,8]}   
randomforest_parameters ={'randomforest__max_features':[0.6,0.8,0.9], 'randomforest__n_estimators':[5,10,15,20],'randomforest__min_samples_leaf':[1,2,4,5] }
svm_parameters = {'svm__C':[1,2,5,10],'svm__gamma':[10,1,0.1,0.001], 'svm__kernel':['rbf','linear','poly']}

cv_decision = GridSearchCV(decisiontree_kbest, param_grid=decision_parameters)
cv_randomforest = GridSearchCV(randomforest_kbest, param_grid=randomforest_parameters)
cv_svm = GridSearchCV(svm_kbest, param_grid=svm_parameters)
'''    

'''
test_classifier(cv_decision, my_dataset, features_list)  

print("Best estimator is ",cv_decision.best_estimator_)
print("Best score is ", cv_decision.best_score_)
 
'''
#Best estimator for Random Forest
'''
test_classifier(cv_randomforest, my_dataset, features_list)  

print("Best estimator is ",cv_randomforest.best_estimator_)
print("Best score is ", cv_randomforest.best_score_)
'''
#Best estimator for SVM
'''
test_classifier(cv_svm, my_dataset, features_list)  

print("Best estimator is ",cv_svm.best_estimator_)
print("Best score is ", cv_svm.best_score_)
'''

#Evaluating DecisionTree Algorithm using tester.py 

test_classifier(decisiontree_kbest, my_dataset, features_list) 
    

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(decisiontree_kbest, my_dataset, features_list)