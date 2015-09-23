# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:20:21 2013

@author: sahill
"""


'''
=====`TRAIN' Usage======
options:
-s type : set type of solver (default 1)
	 0 -- L2-regularized logistic regression (primal)
	 1 -- L2-regularized L2-loss support vector classification (dual)
	 2 -- L2-regularized L2-loss support vector classification (primal)
	 3 -- L2-regularized L1-loss support vector classification (dual)
	 4 -- multi-class support vector classification by Crammer and Singer
	 5 -- L1-regularized L2-loss support vector classification
	 6 -- L1-regularized logistic regression
	 7 -- L2-regularized logistic regression (dual)
	11 -- L2-regularized L2-loss epsilon support vector regression (primal)
	12 -- L2-regularized L2-loss epsilon support vector regression (dual)
	13 -- L2-regularized L1-loss epsilon support vector regression (dual)
-c cost : set the parameter C (default 1)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-e epsilon : set tolerance of termination criterion
	-s 0 and 2
		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
		where f is the primal function and pos/neg are # of
		positive/negative data (default 0.01)
	-s 11
		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001) 
	-s 1, 3, 4 and 7
		Dual maximal violation <= eps; similar to libsvm (default 0.1)
	-s 5 and 6
		|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,
		where f is the primal function (default 0.01)
	-s 12 and 13\n"
		|f'(alpha)|_1 <= eps |f'(alpha0)|,
		where f is the dual function (default 0.1)
-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
-wi weight: weights adjust the parameter C of different classes (see README for details)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)

====`PREDICT' Usage=======

Usage: predict [options] test_file model_file output_file
options:
-b probability_estimates: whether to output probability estimates, 
0 or 1 (default 0); currently for logistic regression only

Note that -b is only needed in the prediction phase. This is different
from the setting of LIBSVM.
'''



import os
import numpy as np
import pandas as pd
from liblinearutil import *
from sklearn.metrics import roc_auc_score,log_loss
from math import log
import time

def run(analysis_path,feature_str_selection,folds,total_folds,train_per):
    #==== Params ====#
    #folds = 5
    selec_criteria = 'auc' # auc or rig
    #===== FILES =====#
    executable_path='../../../lib/liblinear-weights/'
    #data_path='../data/21_1_5F_0C_[\'site_id\']_sequential/'
    #==================#
    
    train_num_arr=train_per
    cut_off_arr=[0]#,10,30,100,300,1000,3000,9000,25000,75000]
    #c_arr= 0.01* np.logspace(0, 3)
    c_arr=[0.37170075843439998]
    tol = 0.00000001
    model = 0
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
    def Create_Folder_Name(str_arr):
        folder_name=''
        for element in str_arr:
            #print element
            folder_name=folder_name+element[0:2]
            if element.find('*')>0:
                folder_name=folder_name+element[element.find('*'):element.find('*')+3]
            folder_name=folder_name+'_'
        return folder_name[0:-1]
    def GetTrainTestNum(data_path):
        #slash_ind=data_path.find('data')
        slash_ind=(data_path.find('_')-data_path[data_path.find('_')-3:-1].find('/'))-2
        first_ind=data_path.find('_')
        second_ind=data_path[first_ind+1:-1].find('_')+1
    
        
        train_num=data_path[slash_ind:first_ind]
        test_num=data_path[first_ind+1:first_ind+second_ind]
        return train_num,test_num
    
    def AppendLogCSV(log_file_name,metric_name,metric_arr,train_num,test_num,cut_off,c,val_arr):
        if not os.path.exists(log_file_name):
            header_str='metric,train_day,test_day,cut_off,best_c'
            for val in val_arr:
               header_str=header_str+','+str(val)
            header_str=header_str+"\n"
            with open(log_file_name, "a") as myfile:
                myfile.write(header_str) 
           
        results_str=str(metric_name)+","+str(train_num)+","+str(test_num)+","+str(cut_off)+","+str(c)
        for m in metric_arr:
           results_str=results_str+','+str(m)       
        results_str=results_str+"\n"
        with open(log_file_name, "a") as myfile:
            myfile.write(results_str) 
    
    
    t0=time.time()
    
    results=dict()
    for tr in train_num_arr:
        for cut_off in cut_off_arr:
            #analysis_path='../data/cutoff_test/'
            data_path=analysis_path+str(tr)+'_1_'+str(total_folds)+'_'+str(cut_off)+'C_'+feature_str_selection
            auc_test_arr=[]
            rig_test_arr=[]
            auc_val_arr=[]
            rig_val_arr=[]
            for k in folds:
                train_file= 'f'+str(k+1)+'_train'
                val_file =  'f'+str(k+1)+'_val'
                test_file=  'f'+str(k+1)+'_test'
                best_val_auc=0
                best_val_rig=0
                for c in c_arr:
                    os.system('./'+executable_path+'train -e '+str(tol)+' -B 1 -s '+str(model)+' -c '+str(c)+' -W '+data_path+train_file+'_samples.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')                                
                    os.system('./'+executable_path+'predict -b 1 '+data_path+val_file+'_svm.txt '+data_path+'model_summ.model '+data_path+'preds_SummModel_py.txt' )                    
                    ##########
                    
                    ##########                    
                    y_val, x_val = svm_read_problem(data_path+val_file+'_svm.txt')               
                    W_val=np.loadtxt(data_path+val_file+'_samples.txt')                
                    results_SummModel=pd.read_csv(data_path+'preds_SummModel_py.txt',sep=' ')
                    p_val1_Summ=results_SummModel['1'].values              
                    y_np=np.array(y_val)
                    W_np=np.array(W_val)
                    y_val_mean=float(W_np[np.where(y_np==1)].sum())/float(W_np.sum())
                    
                    entropy = -(y_val_mean*log(y_val_mean)+(1-y_val_mean)*log(1-y_val_mean))                     
                    log_loss_w=W_LogLos(y_np,p_val1_Summ,W_np)
                    rig_w= 1 - (log_loss_w/entropy)
                    
                    auc_w=W_ROC_AUC(p_val1_Summ,y_np,W_np)
                    
                    auc_val_arr.append(auc_w)
                    rig_val_arr.append(rig_w)
                    print 'Fold '+str(k+1)+' for CV'
                    print 'Cut-Off          :',str(cut_off)
                    print 'C                :',str(c)
                    print 'RIG Weighted     :',rig_w
                    print 'AUC Weighted     :',auc_w
                    print '#########################'
            train_num=str(tr)
            test_num=str(1)
            results['auc_arr']=auc_val_arr
            results['rig_arr']=rig_val_arr
            results['train_period']=train_num
            results['test_period']=test_num
            results['cut_off']=cut_off
            results['C']=c              
            return results
                #AppendLogCSV(analysis_path+'onlyCutOff_s-a-sa-KFold_auc_val_sel_'+selec_criteria+'_log.csv','auc',auc_val_arr,train_num,test_num,cut_off,best_val_c,c_arr)
                #AppendLogCSV(analysis_path+'onlyCutOff_s-a-sa-KFold_rig_val_sel_'+selec_criteria+'_log.csv','rig',rig_val_arr,train_num,test_num,cut_off,best_val_c,c_arr)
                
    #            ############################
    #            print 'Fold '+str(k+1)+' TEST'
    #            print 'Cut-Off               :',str(cut_off)
    #            print 'BEST C                :',best_val_c
    #            print 'BEST RIG Weighted     :',best_val_rig
    #            print 'BEST AUC Weighted     :',best_val_auc
    #            print '#########################'
    #            os.system('./'+executable_path+'train -e '+str(tol)+' -B 1 -s '+str(model)+' -c '+str(best_val_c)+' -W '+data_path+train_file+'_samples.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')
    #            os.system('./'+executable_path+'predict -b 1 '+data_path+test_file+'_svm.txt '+data_path+'model_summ.model '+data_path+'preds_SummModel_test_py.txt' )
    #                
    #            y_test, x_test = svm_read_problem(data_path+test_file+'_svm.txt')            
    #            W_test=np.loadtxt(data_path+test_file+'_samples.txt')            
    #            results_SummModel=pd.read_csv(data_path+'preds_SummModel_test_py.txt',sep=' ')
    #            p_val1_Summ=results_SummModel['1'].values           
    #            y_np=np.array(y_test)
    #            W_np=np.array(W_test)
    #            y_test_mean=float(W_np[np.where(y_np==1)].sum())/float(W_np.sum())
    #            
    #            entropy = -(y_test_mean*log(y_test_mean)+(1-y_test_mean)*log(1-y_test_mean)) 
    #            
    #            log_loss_w=W_LogLos(y_np,p_val1_Summ,W_np)
    #            rig_w= 1 - (log_loss_w/entropy)
    #            
    #            auc_w=W_ROC_AUC(p_val1_Summ,y_np,W_np)
    #            
    #            auc_test_arr.append(auc_w)
    #            rig_test_arr.append(rig_w)
    
    #        train_num=str(tr)                        
    #        test_num=str(1)
    #        AppendLogCSV(analysis_path+'onlyCutOff_s-a-sa-KFold_auc_test_sel_'+selec_criteria+'_log.csv','auc',auc_test_arr,train_num,test_num,cut_off,best_val_c,['f1','f2','f3','f4','f5'])
    #        AppendLogCSV(analysis_path+'onlyCutOff_s-a-sa-KFold_rig_test_sel_'+selec_criteria+'_log.csv','rig',rig_test_arr,train_num,test_num,cut_off,best_val_c,['f1','f2','f3','f4','f5'])
    print '###########################'
    print 'Finished everything {0}'.format(time.time()-t0)
    print '###########################'