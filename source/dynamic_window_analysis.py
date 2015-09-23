# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:11:57 2013

@author: sahill
"""

import divide_data_2
import libSVM_kFold_dynamicWin
import time

def Create_Folder_Name(str_arr):
    folder_name=''
    str_arr=str_arr.split(',')
    for element in str_arr:
        #print element
        folder_name=folder_name+element[0:2]
        if element.find('*')>0:
            folder_name=folder_name+element[element.find('*'):element.find('*')+3]
        folder_name=folder_name+'_'
    return folder_name[0:-1]

whole_file='../data/september_part00'
analysis_folder='../data/dynamic_train_sep_distance_test' #If not, creates the folder for divide_data

granularity='date' #day or date : date is the minimum granularity of the data, in our case hour
total_folds=2
#data_div_train_per=[2,4,9]#,17,25,49,73,169,337]
data_div_train_per=[337,505]
test_per=1

libSVM_folds=range(10)
libSVM_train_per=[2,4,9]#,17,25,49,73,169,337]
clean_start=False
factors_arr=[['site_id']]#,'site_id,ad_id,site_id*ad_id']

for factors in factors_arr:  
    factors_str='si'#Create_Folder_Name(factors)              
                
    divide_data_2.run(whole_file,analysis_folder,factors,data_div_train_per,test_per,total_folds,factors_str,clean_start,granularity)
    print factors

    #results=libSVM_kFold_dynamicWin.run(analysis_folder+'/',factors_str+'_distance_test/',libSVM_folds,total_folds,libSVM_train_per,test_per,granularity)
#
#    results_all=dict()
#    results_all['features']   =factors    
#    results_all['date']       =time.ctime()    
#    results_all['granularity']=granularity
#    results_all['data']       =results       
#    with open('../data/'+analysis_folder+'/stepwise_log', "a") as myfile:
#        myfile.write(str(results_all)+'\n')
    