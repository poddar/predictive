# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:25:58 2013

@author: sahill
"""

import logging
import numpy as np
import pandas as pd


import sys
from time import time
import math
import cPickle as pkl


from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier,SGDRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.cross_validation import KFold

fold_count=5
t0=time()


#=====Log File===
log_file_name="../data/results_log_"+str(t0)+".csv"
header="features,auc_mean,auc_std,rig_mean,rig_std\n"

with open(log_file_name, "a") as myfile:
    myfile.write(header) 



#===LOADING====
data_sparse=joblib.load('../data/mad_sparse.pkl')
data_y_dict=np.load('../data/mad_y.npz')
f = file('../data/sc', 'rb')
sc = pkl.load(f)
f.close()

data_y=data_y_dict['mad_y']

print "Loaded ALL data in %fmin" % ((time() - t0)/60)

print "Dataset Size : ",len(data_y)
print "GT CTR       : ",sc.click_rate_

#clf = SGDRegressor(alpha=0.0001, eta0=0.001, n_iter=150, fit_intercept=False, shuffle=True,verbose=0)
#clf = MultinomialNB(alpha=0.1, fit_prior=True)
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None) # defaults



#scores = cross_validation.cross_val_score(clf, data_sparse, data_y, cv=5)

kf = KFold(len(data_y),n_folds=fold_count,indices=True)



start_features=np.append(sc.start_features,data_sparse.shape[1])

all_fea=list(np.concatenate((np.array(sc.factors),np.array(sc.non_factors))))


auc_model = 0.5
rig_model = 0

full_list       = list(np.concatenate((np.array(sc.factors),np.array(sc.non_factors))))
selected_list   = []
complement      = list(np.concatenate((np.array(sc.factors),np.array(sc.non_factors))))

len_full_list   = len(full_list)

auc_log = []
rig_log = []

mean_pred_log = []
max_pred_log = []

while(1):
    complement        = list(set(full_list) - set(selected_list))
    
    if len(complement) == 0:
            break   
        
    len_selected_list = len(selected_list)
    best_attribute    = ""
    print "All attributes: ", full_list
    print "Complement", complement

    auc_model_old = auc_model
    rig_model_old = rig_model
    
    for attribute in complement: 
        selected_list.append(attribute)
        print "==="+str(fold_count)+"-Fold CV==="
        print selected_list    
               
        selected_indexes=[]
        selected_sparse=[]
        for slc in selected_list:
            ind = all_fea.index(slc)
            pin = start_features[ind]
            pend= start_features[ind+1]
            
            selected_indexes=np.concatenate((selected_indexes,range(pin,pend)))
                        
        selected_sparse=data_sparse[:,selected_indexes]
            
        #====K-Fold Cross Validation=====
        fold_num=1
        fold_auc_scores=[]
        fold_rig_scores=[]
        for train, test in kf:
            x_train=selected_sparse[train,:]
            y_train=data_y[train]
            
            x_test=selected_sparse[test,:]
            y_test=data_y[test]
            
            clf = clf.fit(x_train,y_train)
            
            predictions = clf.predict_proba(x_test)
           
            auc     = roc_auc_score(y_test.astype(int).ravel(), predictions[:,1])
            fold_auc_scores.append(auc)
            
            
            y_test_mean=y_test.mean()

            entropy2 = -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean))
          
            log_loss_preds = predictions
            #log_loss_preds = np.zeros(shape=(len(predictions),2))
            #for i,pred in enumerate(predictions):
            #    log_loss_preds[i] = [1-pred,pred]

            cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
            rig             = 1 + cross_entropy/entropy2            
            fold_rig_scores.append(rig)
            
            print "AUC-RIG SCORE for fold "+str(fold_num)+"/"+str(fold_count)+" : "+str(auc)+"-"+str(rig)             
            fold_num+=1
        
        auc_mean=(np.array(fold_auc_scores)).mean()    
        rig_mean=(np.array(fold_rig_scores)).mean()
        if auc_mean > auc_model:
            if rig_mean > 0:
                auc_model = auc_mean
                rig_model = rig_mean
                best_attribute = attribute
        # Tie breaking
        # if auc == auc_model:
        #     if random.random() > 0.5:
        #         auc_model = auc
        #         rig_model = rig
        #         best_attribute = attribute
        # If we leave this out, the full space is explored => marginally better performance
        if auc_mean < auc_model_old:
            full_list.remove(attribute)                    
        
        results_str=str(selected_list).replace(",","|")+","+str(auc_mean)+","+str(np.array(fold_auc_scores).std())+","+str(rig)+","+str(np.array(fold_rig_scores).std())+"\n"
        with open(log_file_name, "a") as myfile:
            myfile.write(results_str)        
        
        
        selected_list.remove(attribute)
        print "auc:                ", auc_mean
        print "rig:                ", rig_mean
        print "=================="

    if len(best_attribute) > 0:
        selected_list.append(best_attribute)
    
    # Can be part of the while loop?
    if len_selected_list == len(selected_list):
        break
    
    auc_log.append(auc_model)
    rig_log.append(rig_model)
    mean_pred_log.append(np.mean(predictions))
    max_pred_log.append(np.max(predictions))

    print "Best selected_list so far: ", selected_list
    print "auc:                       ", auc_model
    print "rig:                       ", rig_model

print "best model:          ", selected_list
print "AUC:                 ", auc_model
print "RIG:                 ", rig_model
print "AUC log:             ", auc_log
print "RIG log:             ", rig_log
print "Mean prediction log: ", mean_pred_log
print "Max prediction log:  ", max_pred_log


print "done with EVERYTHING in %fmin" % ((time() - t0)/60)
#X_train,X_test,y_train,y_test = train_test_split(mad_sparse, mad_y, test_size=0.33, random_state=42)









