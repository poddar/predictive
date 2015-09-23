# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:20:21 2013

@author: sahill
"""
import os
import numpy as np
import pandas as pd
from liblinearutil import *
from sklearn.metrics import roc_auc_score,log_loss
import time

def W_LogLos(y,predictions,w):
    ll=0.0
    for i in range(len(y)):
        ll = ll + w[i]*y[i]*log(predictions[i]) + w[i]*(1.0-y[i])*log(1.0-predictions[i])
    return  -(1.0/sum(w))*ll

def W_ROC_AUC(p,y,w):
    preds_all=np.ones(w.sum())
    #preds_all_rig=np.ones([w.sum(),w.sum()])
    pin=0
    y_a=np.ones(w.sum())
    for i in range(len(w)):
        pend=pin+w[i]
        preds_all[pin:pend]=p[i]
        #preds_all_rig[pin:pend]=[(1.0-p[i]),p[i]]
        y_a[pin:pend]=y[i]
        pin=pend
    #log_loss(y_a,preds_all_rig)
    return roc_auc_score(y_a, preds_all)#,log_loss(y_a,preds_all_rig)

def GetTrainTestNum(data_path):
    #slash_ind=data_path.find('data')
    slash_ind=(data_path.find('_')-data_path[data_path.find('_')-3:-1].find('/'))-2
    first_ind=data_path.find('_')
    second_ind=data_path[first_ind+1:-1].find('_')+1

    
    train_num=data_path[slash_ind:first_ind]
    test_num=data_path[first_ind+1:first_ind+second_ind]
    return train_num,test_num

def AppendLogCSV(log_file_name,metric_name,metric_arr,train_num,test_num,cut_off):
    results_str=str(metric_name)+","+str(train_num)+","+str(test_num)+","+str(cut_off)
    for m in metric_arr:
       results_str=results_str+','+str(m)
        
    results_str=results_str+","+str(mean(metric_arr))+"\n"
    with open(log_file_name, "a") as myfile:
        myfile.write(results_str) 

t0=time.time()
#==== Params ====#
folds = 5

#===== FILES =====#
executable_path='../../../lib/liblinear-weights/'
#data_path='../data/21_1_5F_0C_[\'site_id\']_sequential/'
#==================#

train_num_arr=[8]
cut_off_arr=[0]#,10,30,100,300,1000,3000,9000,25000,75000]
#c_arr= 0.01* np.logspace(0, 3)

for tr in train_num_arr:
    for cut_off in cut_off_arr:
        analysis_path='../data/cutoff_test/'
        data_path=analysis_path+str(tr)+'_1_5F_'+str(cut_off)+'C_s-a-sa_sequential/'
        auc_arr=[]
        rig_arr=[]
        for k in range(folds):
            train_file='f'+str(k+1)+'_train'
            test_file='f'+str(k+1)+'_test'
            
            os.system('./'+executable_path+'train -e 0.00000001 -B 1 -s 0 -c 1 -W '+data_path+train_file+'_samples.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')
            
            ##########
            print 'Fold '+str(k+1)
            ##########
            
            os.system('./'+executable_path+'predict -b 1 '+data_path+test_file+'_svm.txt '+data_path+'model_summ.model '+data_path+'preds_SummModel_py.txt' )
            
            y_test, x_test = svm_read_problem(data_path+test_file+'_svm.txt')
            
            W_test=np.loadtxt(data_path+test_file+'_samples.txt')
            
            results_SummModel=pd.read_csv(data_path+'preds_SummModel_py.txt',sep=' ')
            p_val1_Summ=results_SummModel['1'].values
            
            y_np=np.array(y_test)
            W_np=np.array(W_test)
            y_test_mean=float(W_np[np.where(y_np==1)].sum())/float(W_np.sum())
            
            entropy = -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean)) 
            
            log_loss_w=W_LogLos(np.array(y_test),np.array(p_val1_Summ),np.array(W_test))
            rig_w= 1 - (log_loss_w/entropy)
            
            auc_w=W_ROC_AUC(np.array(p_val1_Summ),np.array(y_test),np.array(W_test))
            
            ##########
            ##########
            ##########    
            auc_arr.append(auc_w)
            rig_arr.append(rig_w)
            print 'RIG Weighted     :',rig_w
            print 'AUC Weighted     :',auc_w
            print '#########################'
        train_num,test_num=GetTrainTestNum(data_path)
        train_num=str(tr)
        test_num=str(1)
        AppendLogCSV(analysis_path+'gouped_s-a-sa-KFold_auc_log.csv','auc',auc_arr,train_num,test_num,cut_off)
        AppendLogCSV(analysis_path+'gouped_s-a-sa-KFold_rig_log.csv','rig',rig_arr,train_num,test_num,cut_off)
print '###########################'
print 'Finished everything {0}'.format(time.time()-t0)
print '###########################'