# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:11:57 2013

@author: sahill
"""

import divide_data
import libSVM_kFold_stepwise
import time

def Create_Folder_Name(str_arr):
    folder_name=''
    for element in str_arr:
        #print element
        folder_name=folder_name+element[0:2]
        if element.find('*')>0:
            folder_name=folder_name+element[element.find('*'):element.find('*')+3]
        folder_name=folder_name+'_'
    return folder_name[0:-1]

whole_file='../data/september_part00'
analysis_folder='../data/stepwise_analysis_sep' #If not, creates the folder for divide_data

granularity='date' #day or date : date is the minimum granularity of the data, in our case hour
total_folds=5
data_div_train_per=[1,4,8,16,24]
test_per=1

libSVM_folds=[1]
libSVM_train_per=[1,4,8]
clean_start=False
factors_arr=[['site']]
#factors_arr=[['site_id','device','site_id*device']]#,
#factors_arr=[['site_id','banner_type','site_id*banner_type']]#,
#factors_arr=[['site_id','account','site_id*account']]#,
#factors_arr=[['site_id','banner_type','account','site_id*device','site_id*banner_type','site_id*account']]



for factors in factors_arr:  
    factors_str=Create_Folder_Name(factors)              
                
    divide_data.run(whole_file,analysis_folder,factors_arr,data_div_train_per,test_per,total_folds,factors_str,clean_start,granularity)
    print factors

    results=libSVM_kFold_stepwise.run(analysis_folder+'/',factors_str+'_sequential/',libSVM_folds,total_folds,libSVM_train_per)

    results_all=dict()
    results_all['features']   =factors    
    results_all['date']       =time.ctime()    
    results_all['granularity']=granularity
    results_all['data']       =results       
    with open('../data/'+analysis_folder+'/stepwise_log', "a") as myfile:
        myfile.write(str(results_all)+'\n')
    