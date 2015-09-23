# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:04:25 2013

@author: sahill
"""

import time

import numpy as np
import pandas as pd
import gen_features
from sklearn.externals import joblib
from sklearn.datasets import dump_svmlight_file
import cPickle as pkl

data_path='../data/'
data_file='query_dummy.csv'
output_file_name='dummy_mad'

#mad=pd.read_table('../data/train1M_sep.csv',header=True,sep='|',dtype={
#      'ad':object,'account':object,'advertiser_account':object,'age':object,'app_id':object,
#	'app_site':object,'audience':object,'banner_type':object,'campaign':object,'carrier':object,'channel':object,'country':object,'device':object,
#	'gender':object,'geo_target':object,'gps_is_precise':bool,'hour':object,'income':object,'ip':object,'idfa':object,'location':object,'mraid':object,
#	'platform':object,'requester':object,'rtb':bool,'site':object,'user_token':object,'weekday':object,'clicks':int},
#names=['ad','account','advertiser_account','age','app_id','app_site','audience','banner_type','campaign','carrier',
#	   'channel','country','device','gender','geo_target','gps_is_precise','hour','income','ip','idfa','location',
#	   'mraid','platform','requester','rtb','site','user_token','weekday','clicks'])
    


#mad=pd.read_table(data_path+data_file,header=True,sep=',',dtype={
#      'date':object,'ad_id':object,'site_id':object,'click':int,'samples':int},
#names=['date','ad_id','site_id','click','samples'])



#
#
#
#'token':object,	'site':object,	'ad':object,	'country':object,	'rtb':bool,	'acct':object,	
#'campaign':object,	'banner':int,	'spotbuy':object,	'appid':object,	'device':object,	'token1':object,	'clicks':int,	'conversions':int},
#names=['token',	'site',	'ad',	'country',	'rtb',	'acct',	'campaign',	'banner',	'spotbuy',	'appid',	'device',	'token1',	'clicks',	'conversions'])
#names=['ad','account','advertiser_account','age','app_id','app_site','audience','banner_type','campaign','carrier','channel','country','device','gender','geo_target','gps_is_precise','hour',
#       'income','ip','idfa','location','mraid','platform','requester','rtb','site','user_token','weekday','clicks']
    
    
mad=pd.read_csv(data_path+data_file,header=0)

    
    
#mad=pd.read_table('../data/sample.tsv',header=None,dtype={
#'token':object,	'site':object,	'ad':object,	'country':object,	'rtb':bool,	'acct':object,	
#'campaign':object,	'banner':int,	'spotbuy':object,	'appid':object,	'device':object,	'token1':object,	'clicks':int,	'conversions':int},
#names=['token',	'site',	'ad',	'country',	'rtb',	'acct',	'campaign',	'banner',	'spotbuy',	'appid',	'device',	'token1',	'clicks',	'conversions'])


#mad.fillna('NA',inplace=True)
#mad_y=pd.read_table('full.tsv',usecols=[12])
mad_y=mad['clicks'].values.astype(np.int8).squeeze()



factors=['ad_id','site_id','ad_id*site_id']#,'ad*country','ad*device','site*country'
factors=['site_id']#,'ad*country','ad*device','site*country'
non_factors=['views']

sc=gen_features.SparseCat(factors,non_factors)
sc.set_params(count_cutoff=25)
t1=time.time()
sc.fit(mad,mad_y)
t2=time.time()
print (time.time()-t1)/60

# can also do interactions
#factors_cross=['ad','campaign','account','site','country','device','ad*site','ad*country','ad*device','site*country']
#sc_cross=gen_features.SparseCat(factors_cross,non_factors)
#sc_cross.set_params(count_cutoff=25)

mad_sparse=sc.transform(mad)
f = file(data_path+'sc', 'wb')
pkl.dump(sc,f,protocol=pkl.HIGHEST_PROTOCOL)
f.close()

# different ways of saving the sparse matrix - using sklearn
joblib.dump(mad_sparse, data_path+output_file_name+'_sparse.pkl')
# gen_features.save_sparse("mad_sparse",mad_sparse) # or numpy as npz file

# save clicks
np.savez(data_path+output_file_name+'_y',mad_y=mad_y)

mad['samples'].to_csv(data_path+output_file_name+'_samples.csv',header=['sample'])

gen_features.csr_write_libsvm(data_path+output_file_name+'_svm.txt',mad_sparse, mad_y, len(factors)+len(non_factors))
#dump_svmlight_file(mad_sparse, mad_y, '../data/mad_svm.txt', zero_based=True, comment=None, query_id=None)