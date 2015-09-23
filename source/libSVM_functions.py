# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:47:41 2014

@author: sahill
"""




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
from sklearn.metrics import roc_auc_score, log_loss
from liblinearutil import *
from datetime import datetime, timedelta

def subtract_hour(datetime_str,hour):
	mytime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
	mytime -= timedelta(hours=hour)
	return mytime.strftime("%Y-%m-%d %H:%M:%S")

def add_hour(datetime_str,hour):
	mytime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
	mytime += timedelta(hours=hour)
	return mytime.strftime("%Y-%m-%d %H:%M:%S")

def MySQL_getdata(sql_table,train_per,test_per,features):
	#Getting data from MySql
	con = MySQLdb.connect(host="opt0.madvertise.net",user="dse", passwd="27f7b0bc",db="dse")
		
	sql = "SELECT "+', '.join(features)+", SUM(`clicks`) as clicks, SUM(`instances`) as instances FROM `"+sql_table+"` WHERE date in ('"+'\',\''.join(test_per)+"') GROUP BY "+ ', '.join(features)+";"
	test_df = psql.frame_query(sql,con)
		
	sql = "SELECT "+', '.join(features)+", SUM(`clicks`) as clicks, SUM(`instances`) as instances FROM `"+sql_table+"` WHERE date >= '"+train_per[0]+"' AND date <= '"+train_per[1]+"' GROUP BY "+ ', '.join(features)+";"
	train_df = psql.frame_query(sql,con)
	
	con.close()
	
	return train_df,test_df
    
##############################
######-----METRICS-----#######
##############################
def W_ROC_AUC(p,y,w):
	preds_all=np.ones(w.sum())
	pin=0
	y_a=np.ones(w.sum())
	for i in range(len(w)):
		pend=pin+w[i]
		preds_all[pin:pend]=p[i]
		#preds_all_rig[pin:pend]=[(1.0-p[i]),p[i]]
		y_a[pin:pend]=y[i]
		pin=pend
	return roc_auc_score(y_a, preds_all)

def W_LogLos(y,predictions,w):
    ll=0.0
    for i in range(len(y)):
        ll = ll + w[i]*(y[i]*log(predictions[i]) + (1.0-y[i])*log(1.0-predictions[i]))
    return  -(ll/sum(w))

def Array_W_LogLos(y,predictions,w):
    w_total=sum(w)
    log_loss_arr=[]
    for i in range(len(y)):
        log_loss_arr.append(- w[i]*((y[i]*log(predictions[i])) + (1.0-y[i])*log(1.0-predictions[i]))/w_total)
    return  log_loss_arr

def make_sparse(test_df,train_df,factors,non_factors,cut_off,data_path):
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	#######################
	# CREATING SVM FORMAT #
	#######################
	tmp_df  =train_df[:]
	tmp_df_1=tmp_df[:]
	tmp_df_1['click_flag']=1
	tmp_df_1['ais']=tmp_df_1['clicks']
	
	tmp_df_0=tmp_df[:]
	tmp_df_0['click_flag']=0
	tmp_df_0['ais']=tmp_df_1['instances']-tmp_df_1['clicks']
	
	train_df=tmp_df_0.append(tmp_df_1)
	train_df=train_df.drop('clicks',1)
	train_df=train_df.drop('instances',1)
	train_df.rename(columns={'click_flag': 'clicks'}, inplace=True)


	tmp_df  =test_df[:]
	tmp_df_1=tmp_df[:]
	tmp_df_1['click_flag']=1
	tmp_df_1['ais']=tmp_df_1['clicks']
	
	tmp_df_0=tmp_df[:]
	tmp_df_0['click_flag']=0
	tmp_df_0['ais']=tmp_df_1['instances']-tmp_df_1['clicks']
	
	test_df=tmp_df_0.append(tmp_df_1)
	test_df=test_df.drop('clicks',1)
	test_df=test_df.drop('instances',1)
	test_df.rename(columns={'click_flag': 'clicks'}, inplace=True)
	
	sc=gen_features.SparseCat(factors,non_factors)
	sc.set_params(count_cutoff=cut_off)
	sc.fit_weighted(train_df,train_df['ais'])
	
	f = file(data_path+'train_SC', 'wb')
	pkl.dump(sc,f,protocol=pkl.HIGHEST_PROTOCOL)
	f.close()
	mad_sparse_train=sc.transform(train_df)
	np.savetxt(data_path+'train_ais.txt', np.array(train_df['ais']), fmt='%d')
	gen_features.csr_write_libsvm(data_path+'train_svm.txt',mad_sparse_train, train_df['clicks'], len(factors)+len(non_factors))
	
	mad_sparse_test=sc.transform(test_df)
	np.savetxt(data_path+'test_ais.txt', np.array(test_df['ais']), fmt='%d')
	gen_features.csr_write_libsvm(data_path+'test_svm.txt',mad_sparse_test, test_df['clicks'], len(factors)+len(non_factors))
			
	return test_df,train_df

