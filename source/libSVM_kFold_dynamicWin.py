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
import cPickle as pkl

def run(analysis_path,feature_str_selection,folds,total_folds,train_per,test_per,granularity):
    #=====Comment here if .run is used========#
    #    analysis_path='../data/dynamic_train_sep/'
    #    feature_str_selection='si_sequential/'
    #    folds=range(3)
    #    total_folds=5
    #    train_per=[2,4]#[2,4,9,17,25,49,72]
    #    test_per=1
    #    granularity='date'
    #=========================================#
    
    executable_path='../../../lib/liblinear-weights/'
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
        
    def Array_LogLos(y,predictions,w):
        #ll=0.0
        w_total=sum(w)
        log_loss_arr=[]
        for i in range(len(y)):
            log_loss_arr.append(-(w[i]*y[i]*log(predictions[i]) + w[i]*(1.0-y[i])*log(1.0-predictions[i]))/w_total)
            #ll = ll + w[i]*y[i]*log(predictions[i]) + w[i]*(1.0-y[i])*log(1.0-predictions[i])
        return  log_loss_arr
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
    
    
       
    dynamic_train_dict=dict()
    auc_mat=[]
    for tr in train_num_arr:
        #for cut_off in cut_off_arr:
        cut_off=cut_off_arr[0]
        data_path=analysis_path+str(tr)+'_'+str(test_per)+'_'+str(total_folds)+'F_'+str(cut_off)+'C_'+feature_str_selection
        auc_val_arr=[]
        rig_val_arr=[]
        logs_dict =dict() 
        folds_dict=dict()    
        results_dict=dict()
        gp_site=[]
        test_dates_dict=dict()
        train_dates_dict=dict()        
        for k in folds:
            train_file= 'f'+str(k+1)+'_train'
            val_file =  'f'+str(k+1)+'_val'
            test_file=  'f'+str(k+1)+'_test'
            #for c in c_arr:
            c=c_arr[0]
            os.system('./'+executable_path+'train -e '+str(tol)+' -B 1 -s '+str(model)+' -c '+str(c)+' -W '+data_path+train_file+'_samples.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')                                
            os.system('./'+executable_path+'predict -b 1 '+data_path+val_file+'_svm.txt '+data_path+'model_summ.model '+data_path+'preds_SummModel_py.txt' )                    
            ##########
                              
            y_val, x_val     = svm_read_problem(data_path+val_file+'_svm.txt')               
            W_val            =np.loadtxt(data_path+val_file+'_samples.txt')                
            results_SummModel=pd.read_csv(data_path+'preds_SummModel_py.txt',sep=' ')
            p_val1           =results_SummModel['1'].values              
            y_np             =np.array(y_val)
            W_np             =np.array(W_val)
            y_val_mean       =float(W_np[np.where(y_np==1)].sum())/float(W_np.sum())
            
            entropy         = -(y_val_mean*log(y_val_mean)+(1-y_val_mean)*log(1-y_val_mean))                     
            log_loss_w      = W_LogLos(y_np,p_val1,W_np)
            rig_w           = 1 - (log_loss_w/entropy)
            
            auc_w           =W_ROC_AUC(p_val1,y_np,W_np)
            
            f            = file(data_path+train_file+'_SC','rb')
            sc_train     =pkl.load(f)
            f.close()
            fc_tb_train  =sc_train.factors_table_    
            site_df_train=fc_tb_train['site_id']
            site_df_train['mean']=site_df_train['mean']*100

        
            f = file(data_path+val_file+'_SC','rb')
            sc_test=pkl.load(f)
            f.close()    
            fc_tb_test=sc_test.factors_table_    
            site_df_test=fc_tb_test['site_id']
            site_df_test['mean']=site_df_test['mean']*100

                                              
            test_df=pd.read_csv(data_path+val_file+'_df.csv',sep=',')
            #test_df=pd.read_csv(data_path+test_file+'_df.csv',sep=',')
            train_df=pd.read_csv(data_path+train_file+'_df.csv',sep=',')
            
            test_df['preds']=p_val1
            test_df.to_csv(data_path+val_file+'_df.csv',sep=',')
            test_df['preds']=p_val1*100
            log_loss_arr=Array_LogLos(y_np,p_val1,W_np)
            test_df['log_loss']=(np.array(log_loss_arr)/sum(log_loss_arr))*100
            test_df['w_preds']=test_df['samples']*test_df['preds']
            
            ##site_names=pd.read_csv('/Users/tan/dse/projects/ctr-prediction/data/site_id.csv',sep=',')
            gp_site=test_df.groupby('site_id').sum()
            gp_site['pCtr']=gp_site['w_preds']/gp_site['samples']
            site_df_test_srt=site_df_test.sort_index()
            gp_site['testCtr']=site_df_test_srt['mean']
            gp_site['ctr_err']=abs(gp_site['testCtr']-gp_site['pCtr'])
            gp_site = gp_site[['samples','pCtr','testCtr','ctr_err','log_loss']].sort(columns='samples',ascending=False)
            gp_site['imp_ratios']=np.double(gp_site['samples'])/gp_site['samples'].sum()      
            
            gp_site=gp_site.join(site_df_train['count'])
            gp_site=gp_site.fillna(0)
            gp_site=gp_site.join(site_df_train['mean'])
            gp_site=gp_site.fillna(-1)
            gp_site=gp_site.rename(columns={'samples':'impressions_test','count':'impressions_train','mean':'trainCtr'})
  
            folds_dict[str(k+1)]=gp_site  
            test_dates_dict[str(k+1)]=str(test_df.date.unique())                
            train_dates_dict[str(k+1)]=str(train_df.date.unique()) 
            auc_val_arr.append(auc_w)
            rig_val_arr.append(rig_w)
#            print 'Fold '+str(k+1)+' for CV'
#            print 'Train            :',str(tr)
#            print 'Train Date       :',str(train_df.date.unique())
#            print 'Test Date        :',str(test_df.date.unique())
#            print 'Cut-Off          :',str(cut_off)
#            print 'C                :',str(c)
#            print 'RIG Weighted     :',rig_w
            #print 'AUC Weighted     :',auc_w
            #print '#########################'
        train_num=str(tr)
        test_num=str(1)
        logs_dict['auc_arr']=auc_val_arr
        logs_dict['rig_arr']=rig_val_arr
        #logs_dict['test_dates']=str(test_df.date.unique())
        #logs_dict['train_dates']=str(train_df.date.unique())
        logs_dict['train_period']=train_num+'_'+granularity
        logs_dict['test_period']=test_num+'_'+granularity
        logs_dict['cut_off']=cut_off
        logs_dict['C']=c       
        results_dict['folds_dict']=folds_dict
        results_dict['test_dates']=test_dates_dict
        results_dict['train_dates']=train_dates_dict
        results_dict['logs']=logs_dict
        #print logs['auc_arr']
        auc_mat.append(auc_val_arr)
        dynamic_train_dict[str(tr)]=results_dict
    print auc_mat
#    print dynamic_train_dict[str(tr)]['logs']['auc_arr']
#    print dynamic_train_dict['2']['logs']['auc_arr']   
#    print dynamic_train_dict['4']['logs']['auc_arr']
#    print dynamic_train_dict['9']['logs']['auc_arr']
    f = file('../data/all_resutls_dict_'+time.asctime().replace(' ','_').replace(':','_'), 'wb')
    pkl.dump(dynamic_train_dict,f,protocol=pkl.HIGHEST_PROTOCOL)
    f.close()   
    return dynamic_train_dict
    print '###########################'
    print 'Finished everything {0}'.format(time.time()-t0)
    print '###########################'