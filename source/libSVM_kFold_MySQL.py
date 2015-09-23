#from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:47:41 2014

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



import MySQLdb
import pandas.io.sql as psql

import pandas as pd
import numpy as np
import os
import gen_features
import cPickle as pkl
import time
import random
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from liblinearutil import *
from datetime import datetime, timedelta

import libSVM_functions

def get_prob(w0,w):
	return 1/(1+pow(e,-(w0+w)))


test_per=['2014-01-01 00:00:00']
train_per=['2014-01-01 01:00:00','2014-01-15 00:00:00']

sql_fields=['site']#,'salesforce_id','ad_brand','brand_internal_category']
features  =['site']#,'salesforce_id','ad_brand','brand_internal_category']#,'IAB_Category']#,'banner_height','banner_width']



#################################
###  GETING DATA FROM MySQL  ####
#################################
'''Data is aggregated by features'''
sql_table='aggregated_january_all'
train_df,test_df=libSVM_functions.MySQL_getdata(sql_table,train_per,test_per,features)

'''For testing the w values'''
train_df=train_df.sort(columns='instances',ascending=False)[:1]


#################################
###       Sparse Matrix      ####
#################################
factors=features
non_factors=[]
cut_off=0
data_path='../data/aggregated_data/'
executable_path='../../../lib/liblinear-weights/'


'''Sparse matrix is created in datapath'''
test_df_1,train_df_1=libSVM_functions.make_sparse(test_df,train_df,factors,non_factors,cut_off,data_path)


#################################
###          LibSVM          ####
#################################
'''Name of the files'''
train_file='train'
test_file='test'


c=0#0.37170075843439998 #regularization
tol = 0.00000001
model = 0 #model selection, check the header
bias = -1

#os.system('./'+executable_path+'train -e '+str(tol)+' -B '+str(bias)+' -s '+str(model)+' -c '+str(c)+' -W '+data_path+train_file+'_ais.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')  
''' Deleting the Model for safety, if train doesnot work it will uses pre-trained model'''

os.system('rm '+data_path+'model_summ.model')
os.system('./'+executable_path+'train -s 0 -e 0.00000001 -B -1 '+'-W '+data_path+train_file+'_ais.txt '+data_path+train_file+'_svm.txt '+data_path+'model_summ.model')

os.system('./'+executable_path+'predict -b 1 '+data_path+test_file+'_svm.txt '+data_path+'model_summ.model '+data_path+'preds_SummModel_py.txt' ) 


y_val, x_val      = svm_read_problem(data_path+test_file+'_svm.txt')               
W_val             = np.loadtxt(data_path+test_file+'_ais.txt')                
results_SummModel = pd.read_csv(data_path+'preds_SummModel_py.txt',sep=' ')
p_val1            = results_SummModel['1'].values              

auc_w=libSVM_functions.W_ROC_AUC(p_val1,np.array(y_val),np.array(W_val))

test_df_1['preds']=p_val1
train_df_1.to_csv(data_path+'train.csv',sep=',',index=False)
test_df_1.to_csv(data_path+'test.csv',sep=',')



print 'AUC for '+train_file+'-'+test_file+' : '+str(auc_w)











