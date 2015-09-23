# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:46:07 2014

@author: sahill
"""
#from _future_ import division
import pandas as pd
import numpy as np
import os
import gen_features
import cPickle as pkl
import time
import random
from sklearn.externals import joblib

def run(whole_file,analysis_folder,factors_arr,train_per,test_per,folds,factors_str,clean_start,granularity):
    #---WHOLE FILE
    #y_all=pd.read_csv('../data/september_stepwise_all',sep='|')
    y_all=pd.read_csv(whole_file,sep='|')
      
    #nonComp_folder_name = 'shuff_NP_'+select
      
    ####======================####
    ##=----Only Change Here---=##
    ###========================###
    select='distance_test' #selecting how to divide data sequential, random_day, random_all,distance_test
    k=folds
    test_period=test_per   # in days
    val_period=1 # chosen from the train dates
    train_period_arr=train_per#in days
    cut_off_arr=[0]#,10,30,100,300,1000,3000,9000,25000,75000]
    make_csv=True
    distance_all_per=193#selectes this amount of time-hours-
    distance_test_per=24#selects this amount of time-hours- inside the distance_all_per
    distance_fold_count=0#count for distance implementation
    #granularity='day' #day or date : date is by hour
    ####======================####
    ##==========================##
    ###========================###
    
    #factors_arr=['site_id','ad_id','site_id*ad_id']
    #factors_str='s-a-sa'
   
    if not os.path.exists('../data/'+analysis_folder):
        os.makedirs('../data/'+analysis_folder)
    
    #date_df=y_all.groupby('date')
    y_all['day']=[x[:10] for x in y_all['date']]

    dates=list(y_all[granularity].unique())
    dates.sort()
    dates.reverse()
    dates_len=len(dates)

    #y_date=y_all.set_index(['date','ad_id','site_id'])
    def Create_Folder_Name(str_arr):
        folder_name=''
        for element in str_arr:
            #print element
            folder_name=folder_name+element[0:2]
            if element.find('*')>0:
                folder_name=folder_name+element[element.find('*'):element.find('*')+3]
            folder_name=folder_name+'_'
        return folder_name[0:-1]
    
    def make_sparse(df_train,W_train,df_val,W_val,df_test,W_test,file_name,cut_off,factors):
        df_train_y=df_train['clicks'].values.astype(np.int8).squeeze()
        #factors=factors_arr#,'ad_id','site_id*ad_id']#,'ad*country','ad*device','site*country'
        non_factors=[]
        
        sc=gen_features.SparseCat(factors,non_factors)
        sc.set_params(count_cutoff=cut_off)
        sc.fit_weighted(df_train,W_train)
        
        f = file(file_name+'train_SC', 'wb')
        pkl.dump(sc,f,protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
       
        mad_sparse_train=sc.transform(df_train)
        np.savetxt(file_name+'train_samples.txt', np.array(df_train['samples']), fmt='%d')
        gen_features.csr_write_libsvm(file_name+'train_svm.txt',mad_sparse_train, df_train_y, len(factors)+len(non_factors))
        
        if df_val.shape[0] != 0:
            df_val_y=df_val['clicks'].values.astype(np.int8).squeeze()
            mad_sparse_val=sc.transform(df_val)
            np.savetxt(file_name+'val_samples.txt', np.array(df_val['samples']), fmt='%d')
            gen_features.csr_write_libsvm(file_name+'val_svm.txt',mad_sparse_val, df_val_y, len(factors)+len(non_factors))
            sc_val=gen_features.SparseCat(factors,non_factors)
            sc_val.set_params(count_cutoff=cut_off)
            sc_val.fit_weighted(df_val,W_val)
            f = file(file_name+'val_SC', 'wb')
            pkl.dump(sc_val,f,protocol=pkl.HIGHEST_PROTOCOL)
            f.close()
        
        df_test_y=df_test['clicks'].values.astype(np.int8).squeeze()
        mad_sparse_test=sc.transform(df_test)
        np.savetxt(file_name+'test_samples.txt', np.array(df_test['samples']), fmt='%d')
        gen_features.csr_write_libsvm(file_name+'test_svm.txt',mad_sparse_test, df_test_y, len(factors)+len(non_factors))

        sc=gen_features.SparseCat(factors,non_factors)
        sc.set_params(count_cutoff=cut_off)								
        sc.fit_weighted(df_train,W_train)
        sc_test=gen_features.SparseCat(factors,non_factors)
        sc_test.set_params(count_cutoff=cut_off)
        sc_test.fit_weighted(df_test,W_test)
        f = file(file_name+'test_SC', 'wb')
        pkl.dump(sc_test,f,protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
    
    def expand_df_txt(path):
        samples=path+'_test_samples.txt'
        svm_file=path+'_test_svm.txt'
        svm_ex_file=path+'_ex_test_svm.txt'
        y_file=path+'_ex_y.txt'
        W=np.loadtxt(samples)
        spr = open(svm_file, "r") 
        spr_ex= open(svm_ex_file,'w')
        y_ex= open(y_file,'w')
        line_index=0
        for raw in spr:
            for i in range(int(W[line_index])):
                spr_ex.write(raw)
                y_ex.write(raw[0]+'\n')
            line_index+=1
        spr.close()
        spr_ex.close()
    for prd in train_period_arr:
        for cut_off in cut_off_arr:
            train_period=prd
            folder_name = analysis_folder+'/'+str(train_period)+'_'+str(test_period)+'_'+str(k)+'F_'+str(cut_off)+'C_'+factors_str+'_'+select  
            if clean_start:
                if not os.path.exists('../data/'+folder_name):
                    os.makedirs('../data/'+folder_name)
                    print 'Folder Not Exists, Creating the Folder and Dividing Data'
                print 'Overwriting the Folder '+'../data/'+folder_name
            else:
                if not os.path.exists('../data/'+folder_name):
                    os.makedirs('../data/'+folder_name)
                    print '../data/'+folder_name+' Not Exists, Creating the Folder and Dividing Data'
                else:
                    print '../data/'+folder_name+' Already Exists, --Using Existing Folder--'
                    return
            
            for i in range(k):
                if select =='sequential':
                    pin_test   = (i*test_period)
                    pend_test  = pin_test+test_period
                    test_dates = dates[pin_test:pend_test]
                    val_dates  = dates[pend_test:pend_test+val_period]
                    train_dates= dates[pend_test+val_period:pend_test+train_period]
                    print'test_dates', test_dates
                    print 'val_dates', val_dates
                    print'train_dates', train_dates
                    print 'Saving'
                    #== Chossing the portion ==
                    test_df =   y_all[y_all[granularity].isin(test_dates)].reset_index()
                    val_df  =   y_all[y_all[granularity].isin(val_dates)].reset_index()
                    train_df=   y_all[y_all[granularity].isin(train_dates)].reset_index()
                    
                if select == 'random_day':
                    pend=random.randint((test_period+train_period),dates_len)
                    pin=pend-(test_period+train_period)
                    selected_dates_all=dates[pin:pend]
                    test_dates_ind=random.sample(range(0,len(selected_dates_all)),test_period)
                    test_dates=np.array(selected_dates_all)[test_dates_ind]
                    train_dates=np.delete(np.array(selected_dates_all),test_dates_ind)
                    val_dates_ind=random.sample(range(0,len(train_dates)),val_period)
                    val_dates=np.array(train_dates)[val_dates_ind]
                    train_dates=np.delete(np.array(train_dates),val_dates_ind)
                    print'test_dates',test_dates
                    print 'val_dates',val_dates
                    print'train_dates',train_dates
                    #== Chossing the portion ==
                    test_df =   y_all[y_all[granularity].isin(test_dates)].reset_index()
                    val_df  =   y_all[y_all[granularity].isin(val_dates)].reset_index()
                    train_df=   y_all[y_all[granularity].isin(train_dates)].reset_index()
                    
                if select == 'random_all':
                    pend=random.randint((test_period+train_period),dates_len)
                    pin=pend-(test_period+train_period)
                    selected_dates_all=dates[pin:pend]
                    
                    all_df=y_all[y_all[granularity].isin(selected_dates_all)].reset_index()
                    test_len=int(all_df.shape[0]*(test_period/float(train_period)))
                    all_rows=np.array(range(0,all_df.shape[0]))
                    test_rows=random.sample(range(all_df.shape[0]),test_len)    
                    train_rows=np.delete(all_rows,test_rows)
                    #== Chossing the portion ==
                    test_df =   all_df.irow(test_rows)
                    train_df=   all_df.irow(train_rows)
                
                if select == 'distance_test':
                    pin=i*distance_all_per
                    pend=distance_all_per*(i+1)
                    selected_dates_all=dates[pin:pend]

                    selected_dates_test =selected_dates_all[0:distance_test_per]
                    selected_dates_train=selected_dates_all[distance_test_per:pend]

                    for f_i in range(distance_test_per):
                        test_df =   y_all[y_all[granularity].isin([selected_dates_test[f_i]])].reset_index()
                        train_df=   y_all[y_all[granularity].isin(selected_dates_train)].reset_index()
                        val_df  =   y_all[y_all[granularity].isin([])].reset_index()
                        W_val   =   []#val_df['samples']
                        W_train =   train_df['samples'] 
                        W_test  =   test_df['samples']
                        distance_fold_count=distance_fold_count+1
                        make_sparse(train_df,W_train,val_df,W_val,test_df,W_test,'../data/'+folder_name+'/f'+str(distance_fold_count)+'_', cut_off,factors_arr)
                        train_df.to_csv('../data/'+folder_name+'/f'+str(i+1)+'_train_df.csv',sep=',',index=False)
                        test_df.to_csv('../data/'+folder_name+'/f'+str(i+1)+'_test_df.csv',sep=',',index=False)
                    
                if make_csv and select != 'distance_test':
                    train_df.to_csv('../data/'+folder_name+'/f'+str(i+1)+'_train_df.csv',sep=',',index=False)
                    val_df.to_csv('../data/'+folder_name+'/f'+str(i+1)+'_val_df.csv',sep=',',index=False)
                    test_df.to_csv('../data/'+folder_name+'/f'+str(i+1)+'_test_df.csv',sep=',',index=False)
                
                if select != 'distance_test':    
                    W_train = train_df['samples'] 
                    W_val   = val_df['samples']
                    W_test  = test_df['samples']
                    make_sparse(train_df,W_train,val_df,W_val,test_df,W_test,'../data/'+folder_name+'/f'+str(i+1)+'_', cut_off,factors_arr)
                
            


    
    

