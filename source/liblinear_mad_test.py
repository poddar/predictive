# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:55:55 2013

@author: sahill
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:54:24 2013

@author: sean
"""
import time

from sklearn.externals import joblib
from liblinearutil import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

import pandas as pd

data_path='../data/'

# Read data in LIBSVM format
y, x = svm_read_problem(data_path+'mad_svm.txt')
#mad_sparse=joblib.load(data_path+'mad_sparse.pkl')
#mad_y_dict=np.load(data_path+'/mad_y.npz')
#mad_y=mad_y_dict['mad_y']


data_path='../data/dummy_'
t0=time.time()
t1=time.time()
print 'Let the Party Starts...'
# Read data in LIBSVM format
y, x = svm_read_problem(data_path+'_svm.txt')
mad_sparse=joblib.load(data_path+'_sparse.pkl')
mad_y_dict=np.load(data_path+'_y.npz')
mad_y=mad_y_dict['mad_y']

print 'Data is loaded in {0}'.format(time.time()-t1)

# Construct problem in python format
# Dense data
#y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
#y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]

#W = [1] * 200
# W[0] = 1


samples=pd.read_csv(data_path+'samples.csv',header=0)



print "startng prob"
t1=time.time()
W=list(samples['sample'].values) # set W if you want to use weights
prob = problem(W, y, x)
print "Finished prob in {0}".format(time.time()-t1)

# set c
# logistic regression l2 min
# use intercept B=1
c=1
param = parameter('-e 0.0001 -B 1 -c {0} -s 0'.format(c))
#m = train([], y, x, '-c 5')
#m = train(W, y, x)


t1=time.time()

m = train(prob, param)
t2=time.time()
print "Trained liblinear in {0} secs".format(t2-t1)  
 

#lr=LogisticRegression(C=c)
 
#t1=time.time()
#lr.fit(mad_sparse,mad_y)
#t2=time.time()
#print "Trained sklearn in {0} secs".format(t2-t1)  


# cross validation
# not so useful, since only does classification
#CV_ACC = train(W, y[:200], x[:200], '-v 3')
#print CV_ACC,"CVACC"

# they both outpuc class 0 prob and class 1 probs
p_label, p_acc, p_val = predict(y, x, m, '-b 1')
p_val1=[p[1] for p in p_val]

#pred=lr.predict_proba(mad_sparse)[:,1]
#print 'performance on training set'

#auc_sk=metrics.roc_auc_score(y,pred)
auc_svm=metrics.roc_auc_score(y,p_val1)

#print 'auc sk',auc_sk
print 'auc svm',auc_svm
# Other utility functions
save_model('mad_svm.model', m)
#m = load_model('heart_scale.model')
# output prob estimates - b1
#p_label, p_acc, p_val = predict(y, x, m, '-b 1')
#ACC, MSE, SCC = evaluations(y, p_label)

# Getting online help
#help(train)

print 'Finished everything {0}'.format(time.time()-t0)
