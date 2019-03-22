# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:57:34 2018

@author: Vijin
"""

def fraction_from_poi_feature(my_dataset):
    for key,value in my_dataset.iteritems():
        for k, v in value.iteritems():
            if k=="from_poi_to_this_person":
                from_poi_to_this_person=v
            if k=="to_messages":
                to_messages=v         
            
        fraction_from_poi = 0.
           
        if(to_messages>0 and to_messages != 'NaN' and from_poi_to_this_person>0 and from_poi_to_this_person !='NaN'):
            fraction_from_poi = float(from_poi_to_this_person)/float(to_messages)
            fraction_from_poi = round(fraction_from_poi,2)        
        
        my_dataset[key]["fraction_from_poi"] = fraction_from_poi
        
    return my_dataset 
    
        
def fraction_to_poi_feature(my_dataset):        
    for key,value in my_dataset.iteritems():
        for k, v in value.iteritems():
            if k=="from_this_person_to_poi":
                from_this_person_to_poi=v
            if k=="from_messages":
                from_messages=v
            
        fraction_to_poi = 0.   
    
        if(from_messages>0 and from_messages != 'NaN' and from_this_person_to_poi>0 and from_this_person_to_poi !='NaN'):
            fraction_to_poi = float(from_this_person_to_poi)/float(from_messages)
            fraction_to_poi = round(fraction_to_poi,2) 
            
        my_dataset[key]["fraction_to_poi"] = fraction_to_poi
        
    return my_dataset