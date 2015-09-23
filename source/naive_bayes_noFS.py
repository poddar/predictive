# Take in 2 csv files: train and test
# Usage: python bias_model.py <input_train_file> <input_test_file> <output_predictions_file> --delim='delim'

# Load both files into dictionaries and the target variable into an array
# Train the SGD OLS on the train set
# Predict on the test set

# Important change: to make this into a bias model (versus standard SGD OLS),
# you have to remove the mean. Then in making predictions, you add in the mean
# again.

import csv
import sys
import numpy as np
import argparse
import random # we just need random.random
import math # we just need log
import pandas as pd
from   scipy import sparse
from   time   import time
from   pprint import pprint
from   copy   import copy


from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.metrics            import roc_auc_score, mean_absolute_error,log_loss,mean_squared_error
from sklearn.linear_model       import SGDRegressor
from sklearn.naive_bayes        import BernoulliNB,MultinomialNB
from sklearn.isotonic           import IsotonicRegression
from sklearn.externals          import joblib
from sklearn.grid_search        import GridSearchCV
from sklearn.preprocessing      import StandardScaler


######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
def main(argv):
    ############### READ TRAINING FILE ###############
    args             = parse_args()
    input_train_file = args.input_train_file
    input_test_file  = args.input_test_file
    output_file      = args.output_file
    delim            = args.delim
    threshold        = args.threshold
    selected_list   = []
    selected_list   = ['site']
    #selected_list   = ['ad', 'account','advertiser_account', 'app_id', 'campaign', 'country', 'device', 'hour', 'rtb', 'site', 'weekday']

    t100 = time()

    print "Starting!"

    print "Threshold: ", threshold

    ############### READ TEST AND TRAIN FILES ###############
    train     = file_to_dict(input_train_file,delim)
    test      = file_to_dict(input_test_file,delim)
    
    ############### SPLIT DATA FROM TARGETS ###############

    y_train, train = split_target(train)
    y_test, test   = split_target(test)

    print input_train_file+" has "+ str(len(y_train))+" rows."
    print input_test_file+" has "+ str(len(y_test))+" rows."

        # Should we use the built-in normalization functions of Scikit-learn?
        # http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        # from sklearn import preprocessing
        # X_train_scaled = preprocessing.scale(X_train)

        # Question: how to transform back predictions to the proper scale again?
        # Use StandardScaler.inverse_transform() or sth?
        # This is why in SGDRegressor_pred() it is added y_train_mean!!!
    y_train_mean       = np.mean(y_train)
    y_test_mean        = np.mean(y_test)
    y_train_normalized = y_train - y_train_mean


        # entropy = -(y_train_mean*math.log(y_train_mean)+(1-y_train_mean)*math.log(1-y_train_mean))
         
    if y_test_mean==0:
        print "WARNING - zero mean"
        entropy = -(0+(1-y_test_mean)*math.log(1-y_test_mean))
    else:
        entropy = -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean))
    print "entropy: ", entropy
    print "GTA train: ", y_train_mean
    print "GTA test:  ", y_test_mean
    
    auc_model = 0.5
    # relative information gain
    # NB rig will never be zero due to epsilon clipping.# Be careful when deciding whether to keep/discard variables on the basis of having a marginally higher rig!
    # auc: 0.5
    # rig: 0.000962618510464


    log_loss_preds = np.zeros(shape=(len(y_test),2))
    for i in range(0,len(y_test)-1):
        log_loss_preds[i] = [1-y_train_mean,y_train_mean]
    cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
    rig_test        = 1 + cross_entropy/entropy

    print "Cross-entropy: ", cross_entropy
    print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test

    entropy2 = -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean))
    print "entropy2:   ", entropy2
    log_loss_preds2 = np.zeros(shape=(len(y_test),2))
    for i in range(0,len(y_test)-1):
        log_loss_preds2[i] = [1-y_test_mean,y_test_mean]
    
    sum1 = 0


    rig_model = 0
    full_list       = header(input_train_file,delim) 
    complement      = header(input_train_file,delim)

    len_full_list = len(full_list)

    auc_log = []
    rig_log = []
    mean_pred_log = []
    max_pred_log = []

    # If an attribute has only 1 value or all missing (auc=0.5), then remove it from the model / selection process
    # If an attrbitute deteriorates performance, shall we simply remove it from the full list?


    #best AUC-['platform', 'requester', 'idfa', 'weekday']
    #best RIG-['campaign', 'platform', 'requester', 'site', 'country']

    print selected_list

    new_train = []
    new_test  = []

    print "Creating the dictionaries."
    for row in train:
       new_train.append(dict((k, row[k]) for k in selected_list))

    for row in test:
       new_test.append(dict((k, row[k]) for k in selected_list))

    print "Dictionaries are done."
    print "Creating vectorization."
    X_train, X_test = vectorize_dicts(new_train,new_test,threshold)
    print "Vectorization is done."
    if X_train.getnnz() > 0:
        #SGDRegressor_grid_pred(X_train, X_test, y_train_normalized, y_train_mean)
        print "Running Multinomial Naive Bayes." 
        predictions,predictions_train     = MultinomialNB_pred(X_train, X_test, y_train)   
        print "Computing AUC score."

        mse     = mean_squared_error(y_test,predictions)
        auc     = roc_auc_score(y_test.astype(int).ravel(), predictions) # put y_test.astype(int).ravel() in a variable outside the loop   
        print "Computing RIG score."
        # Can be done prettier
        log_loss_preds = np.zeros(shape=(len(predictions),2))
        for i,pred in enumerate(predictions):
            log_loss_preds[i] = [1-pred,pred]    
        
        # print "type(log_loss_preds): ", type(log_loss_preds)
        # print "log_loss_preds.shape: ", log_loss_preds.shape    
        # Warning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
        # Ignore the warning
        
        cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
        rig             = 1 + cross_entropy/entropy2
    else:
         predictions     = np.zeros(1) # hack
         mse             = 0
         auc             = 0.5
         rig             = 0 # rig_mode

    print "Running Isotonic Regression."     
    predictions_ir,predictios_ir_train,params,score_test_ir     = IsotonicRegression_pred(y_train,predictions_train,predictions,1000,y_test)
    print "Computing AUC score." 
    auc_ir                                                      = roc_auc_score(y_test.astype(int).ravel(), predictions_ir) # put y_test.astype(int).ravel() in a variable outside the loop
    print "Computing RIG score." 
    log_loss_preds_ir = np.zeros(shape=(len(predictions_ir),2))
    for i,pred_ir in enumerate(predictions_ir):
        log_loss_preds_ir[i] = [1-pred_ir,pred_ir]

    cross_entropy_ir   = -log_loss(y_test.astype(int).ravel(), log_loss_preds_ir)
    rig_ir             = 1 + cross_entropy_ir/entropy
    ######------TRAIN------######
    # model = 'BIAS '
    # plt.figure(2).clf()
    # plt.subplot(121)
    # Plot_pred_bin(np.array(y_train).ravel(),np.array(predictions_train).ravel(),1001,model+' Model without Isotonic Regression')
    # plt.subplot(122)
    # Plot_pred_bin(np.array(y_train).ravel(),np.array(predictios_ir_train).ravel(),1001,model+' Model with Isotonic Regression')

    # plot_label='Train Size: '+str(len(y_train))+'- Test Size: '+str(len(y_test))+'- TRAIN RESULTS'
    # plt.xlabel(plot_label)

    ######------TEST------######
    # plt.figure(3).clf()
    # plt.subplot(121)
    # Plot_pred_bin(np.array(y_test).ravel(),np.array(predictions).ravel(),1001,model+' Model without Isotonic Regression')
    # plt.subplot(122)
    # Plot_pred_bin(np.array(y_test).ravel(),np.array(predictions_ir).ravel(),1001,model+' Model with Isotonic Regression')

    # plot_label='Train Size: '+str(len(y_train))+'- Test Size: '+str(len(y_test))+'- TEST RESULTS'
    # plt.xlabel(plot_label)

    print "==========================================================================="
    print "Train Size:          ", len(y_train)
    print "Test Size:           ", len(y_test)
    print "best model:          ", selected_list
    print "threshold:           ", threshold
    print "=========================================="
    print "MSE:                 ", mean_squared_error(y_test,predictions)
    print "MSE_IsoReg:          ", mean_squared_error(y_test,predictions_ir)
    print "=========================================="
    print "AUC:                 ", auc
    print "AUC_IsoReg:          ", auc_ir
    print "=========================================="
    print "RIG:                 ", rig
    print "RIG_IsoReg:          ", rig_ir
    print "=========================================="
    print "SCORE:               ", Get_score(predictions,y_test)
    print "SCORE_IsoReg:        ", Get_score(predictions_ir,y_test)
    print "=========================================="


    ############### SAVE TO DATAFRAME and WRITING TO CSV PREDICTIONS ###############
    print "Creating .csv files."
    time_stamp=time()
    d={'y_test':np.array(y_test).ravel(),'Predictions':np.array(predictions).ravel(),'predSIR':np.array(predictions_ir).ravel()}
    df=pd.DataFrame(d) 
    df.to_csv("test_preds_NB_"+str(threshold)+"_"+str(time_stamp)+".csv")

    d1={'y_train':np.array(y_train).ravel(),'Predictions':np.array(predictions_train).ravel()}
    df1=pd.DataFrame(d1)
    df1.to_csv("train_preds_NB_"+str(threshold)+"_"+str(time_stamp)+".csv")

    ############### SAVE THE PREDICTIONS ###############
    print "Saving predictions to file."
    predictions_to_file(predictions,input_test_file,output_file,delim)

    ############### SAVE THE MODEL ###############

    # To do: put this as a command line argument?
    # Something is going wrong here when we do a grid search, so commenting it out
    # joblib.dump(clf, 'bias_model.pkl') 

    ######################################

    print "done with EVERYTHING in %fmin" % ((time() - t100)/60)

    ############### SAVE THE MODEL ###############
    total_time=((time() - t100)/60)

    ############### APPENDING TO LOG FILE ###############
    print "Appending to the log file."
    Append_2_log(time(),total_time,len(y_train),len(y_test),threshold,auc,auc_ir,rig,rig_ir,selected_list," ")


    # Give the user extra command line arguments
    ######------PRINTING------######
    ##plt.figure(1)
    ##plt.plot(np.sort(predictions)) 
    ##plt.plot(np.array(y_test)[np.array(predictions).argsort()],'r')
    
    #plt.figure(2)
    #plt.plot(np.sort(predictions))
    #plt.plot(np.array(predictions_ir)[np.array(predictions).argsort()],'r')

    #sorted_y_test=np.array(y_test)[np.array(predictions).argsort()]
    #prob_target=GetTargetProb(sorted_y_test,1000)
    
    ##print sorted_y_test
    ##print prob_target

    #plt.plot(prob_target,'g')
    #plt.title("Data Length :"+str(len(y_test)))
    #plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Bias model for CTR estimation")

    parser.add_argument("input_train_file",     type=str, help="the input train file")
    parser.add_argument("input_test_file",      type=str, help="the input test file")
    parser.add_argument("output_file",          type=str, help="the output file with predictions")
    parser.add_argument("--delim", default=',', type=str, help="the delimiter in the train, test, and predictions file")
    parser.add_argument("threshold", default=0, type=int, help="threshold for the dictionary")

    args = parser.parse_args()
    # Is this return necessary in Python?
    return args

# TO DO: put all methods below into a separate file for reuse in other models

def predictions_to_file(predictions,input_test_file,output_file,delim):
    """ Would maybe be nice to output sorted on keys, or clicks and prediction as the last columns
    In general this is an ugly function, can we rewrite this?
    """
    test_copy = file_to_dict(input_test_file,delim)
    i = 0
    data_predictions = []

    for row in test_copy:
        a = row.values()
        a.append(predictions[i])
        data_predictions.append(a)
        i += 1

    resultFile = open(output_file,'wb')
    wr = csv.writer(resultFile, delimiter=delim)
    header = test_copy[0].keys()
    header.append('prediction')
    wr.writerow(header)
    wr.writerows(data_predictions)

# Converting to dense numpy arrays causes memory errors
def vectorize_dicts(train,test,threshold=0):
    """ Vectorize dictionaries with cardinality threshold
    """
    vec     = DictVectorizer()
    X_train = vec.fit_transform(train)
    X_test  = vec.transform(test)

    cardinality       = X_train.sum(axis=0)
    cardinality_array = np.asarray(cardinality).flatten()

    selected_features = []
    
    for i,item in enumerate(cardinality_array):
        if item > threshold:
            selected_features.append(i)
    # Change into "ones"?
    # Ugly, but works
    # selected_features = np.zeros(shape=(len(cardinality_array)),dtype=bool)

    # # This loop and the non-zero method can be rolled into 1 if we want to
    # for i,item in enumerate(cardinality_array):
    #     selected_features[i] = False if item < threshold else True

    # selected_features_sparse = selected_features.nonzero()[0]

    # selected_features_sparse = np.array(selected_features)

    if len(selected_features) > 0:
        X_train_selected = X_train[:,selected_features]
        X_test_selected  = X_test[:,selected_features]
    else:
        X_train_selected = sparse.csr_matrix([0]) # Hack to get zero-length matrices
        X_test_selected  = sparse.csr_matrix([0])

    # if selected_features == ['user_token']:
    #     print len(selected_features)


    return X_train_selected, X_test_selected

def file_to_dict(filename,delim):
    """ Method for reading in train or test data
    """
    dict_array = []
    
    # with codecs.open(filename, 'rb', encoding='utf-8') as file_to_read: # new
    with open(filename, 'rb') as file_to_read:
        dicts = csv.DictReader(file_to_read, delimiter=delim)
        for row in dicts:
            dict_array.append(row)
    
    return dict_array

def header(filename,delim):
    """ Method for getting the header as an array
    """
    with open(filename, 'rb') as file_to_read:
        r = csv.reader(file_to_read, delimiter=delim)
        header = r.next()

    header.remove('clicks') # delete this line of clicks also needs to be returned

    return header

def split_target(array_of_dicts):
    length = len(array_of_dicts)
    y = np.zeros(shape=(length,1))

    for index in range(length):
        y[index]=float(array_of_dicts[index]['clicks'])
        del array_of_dicts[index]['clicks']
        # y[index]=float(array_of_dicts[index][u'clicks']) # added in the u
        # del array_of_dicts[index][unicode('clicks',"UTF-8")] # added in the u

    return y, array_of_dicts

def Get_score(predictions,y_test):
    predictions_np = np.array(predictions,float)
    y_test_np = np.array(y_test,float) 

    u_pre=y_test_np.ravel()-predictions_np.ravel()

    u=np.array([u_pre*u_pre],float).sum()

    #print "u", u
    v_pre=y_test_np.ravel()-np.average(y_test_np)
    #print v_pre
    v=np.array([v_pre*v_pre],float).sum()
    #print "v", v
    scr=1-(u/v)
    return scr  


def MultinomialNB_pred(X_train, X_test, y_train):
    clf = MultinomialNB(alpha=0.1, fit_prior=True)
    clf = clf.fit(X_train,y_train)
    
    
    predictions = clf.predict_proba(X_test) # these are predictions for both classes, so non-clicks and clicks
    
    # Get only the predictions for clicks
    predictions_click = []
    for pred in predictions:
        predictions_click.append(pred[1])


    predictions_train = clf.predict_proba(X_train)
    predictions_train_click = []
    for pred in predictions_train:
        predictions_train_click.append(pred[1])

    return predictions_click,predictions_train_click


def SGDRegressor_pred(X_train, X_test, y_train_normalized, y_train_mean,y_test):
    #The learning rate: 
    #---constant: eta = eta0 [assign to the initial one, eta0]
    #---optimal: eta = 1.0/(t+t0) 
    #---invscaling: eta = eta0 / pow(t, power_t) [default]
    clf = SGDRegressor(alpha=0.0001, eta0=0.001, n_iter=150, fit_intercept=False, shuffle=True,verbose=0)
    clf = clf.fit(X_train,y_train_normalized)

    #Conveting to back, (could be used sklearn standardization function for both decoding and encoding)
    predictions_train=clf.predict(X_train) + y_train_mean
    predictions = clf.predict(X_test) + y_train_mean

    score_test=clf.score(X_test,y_test)

    return predictions,predictions_train,score_test

def BernoulliNB_pred(X_train, X_test, y_train):
    clf_NB = BernoulliNB()
    clf_NB.fit(X_train,y_train)

    #Conveting to back, (could be used sklearn standardization function for both decoding and encoding)
    predictions_train=clf_NB.predict_proba(X_train) 
    predictions = clf_NB.predict_proba(X_test) 

    return predictions[:,1],predictions_train[:,1]

def GetTargetProb(y_data_n,bin_step):
    y_data=np.array(y_data_n)
    indexes=np.arange(0,len(y_data),bin_step)
    prob_target=np.zeros(len(y_data))
    for i in range(len(indexes)-1):
        pin =indexes[i]
        pend=indexes[i+1]-1

        prob_target[pin:pend]=np.average(y_data[pin:pend])

    if indexes[-1]<len(y_data):
        pin = indexes[-1]
        pend = len(y_data)
        prob_target[pin:pend] = np.average(y_data[pin:pend])
    return prob_target

def Plot_pred_bin(target,preds,bin_size,plot_name):
    bins=np.linspace(0,1,bin_size)

    #preds_sorted                = np.sort(preds) 
    #target_sorted_by_preds      = target[preds.argsort()]

    d_pred={'target':target,'preds':preds}
    df_pred=pd.DataFrame(d_pred)
    print 'bins',bins
    bin_index_preds     =   np.searchsorted(bins,df_pred['preds'])

    df_pred['bin_index']=   bin_index_preds*(bins[1]-bins[0])
    pred_group          =   df_pred.groupby('bin_index')

    pred_hist           =   pred_group['target'].mean()
    pred_stderr         =   pred_group['target'].std()/np.sqrt(pred_group.size())
    
    plt.axis([0, max(pred_hist.index), 0, max(pred_hist.values)])
    plt.errorbar(pred_hist.index,pred_hist.values,yerr=pred_stderr)
    #plt.plot(pred_hist.index,pred_hist.values)
    plt.plot(bins,bins,'r-')
    #plt.axis('equal')
    plt.title(plot_name)
    axis([0,0.05,0,0.05])
    #plt.xlim(0,max(pred_hist.index))
    #plt.ylim(0,max(pred_hist.values))
    xlabel_str='Prediction Probabilities with total of '+str(bin_size)+ 'bins'
    plt.xlabel(xlabel_str)
    plt.ylabel('Target Probabilities ')
    #preds_sorted                = np.sort(preds)    
    #target_sorted_by_preds      = target[preds.argsort()]

    #np.searchsorted(a, v, side='left', sorter=None)[source]

def IsotonicRegression_pred(y_train,predictions_train,test_preds,bin_step,y_test): 
    # Y Training Target sort the y_test 
    # X Training Data use the indexes of sorted(y_test)
    #y_train_len=len(y_train)
    
    # if bin_step<1:
    #     step_count = 1/bin_step
    # else:
    #     step_count = int(math.floor(y_train_len/bin_step))

    # step_element_count = int(math.floor(y_train_len/step_count))

    # bin_start_indexes=np.array(range(0,step_count))*step_element_count


    predictions_np                      =   np.array(predictions_train,float)
    predictions_sorted                  =   np.sort(predictions_np)
    predictions_sorted_indexes          =   predictions_np.argsort()
    
    y_train_arranged                    =   np.array(y_train,float)[predictions_sorted_indexes].ravel()   
    #not_binned_y_train_arranged         =   y_train_arranged[:]

    # for index in range(len(bin_start_indexes)-1):
    #     pin  = bin_start_indexes[index]
    #     pend = bin_start_indexes[index+1]
    #     y_train_arranged[pin:pend] = np.average(y_train_arranged[pin:pend])
    # if bin_start_indexes[-1]<y_train_len:
    #     pin  = bin_start_indexes[-1]
    #     pend = y_train_len
    #     y_train_arranged[pin:pend] = np.average(y_train_arranged[pin:pend])

    ir          =   IsotonicRegression()

    y_ir        =   ir.fit_transform(predictions_sorted,y_train_arranged)
    y_ir_pred   =   ir.predict(predictions_sorted)


    #print "min(y_train_arranged)    :",    min(y_train_arranged)
    #print "max(y_train_arranged)    :",    max(y_train_arranged)
    #print "min(predictions_sorted)  :",    min(predictions_sorted)
    #print "max(predictions_sorted)  :",    max(predictions_sorted)
    #print "min(test_preds)          :",    min(test_preds) 
    #print "max(test_preds)          :",    max(test_preds)
    #if max(test_preds)>=max(y_train_arranged):
    #np.arrya(test_preds>max(y_train_arranged))==True
    
    max_indexes=np.array((np.where(test_preds>max(y_train_arranged))),int).ravel()
    if len(max_indexes)!=0:
        for m_i in max_indexes:
            test_preds[m_i]=max(y_train_arranged)

    test_preds_sorted        =   np.sort(np.array(test_preds))
    
    predictions_ir           =   ir.predict(test_preds)

    ind = np.where(np.isnan(predictions_ir))[0]
    preds_test_min= np.nanmin(predictions_ir)
    if len(ind)!=0:
        for i in ind:
            predictions_ir[i]=preds_test_min

    
    #==============WRITING TO CSV================
    # d_train={'y_train'          :np.array(y_train,float)[predictions_sorted_indexes].ravel(), 
    #          'y_train_bin'      :np.array(y_train_arranged).ravel(),
    #          'train_preds'      :np.array(predictions_sorted).ravel(),
    #          'train_preds_ir'   :y_ir}

    # df_train=pd.DataFrame(d_train) 
    # df_train.to_csv("train_IR.csv")

    # d_test={'y_test'            :np.array(y_test).ravel(),
    #         'test_preds'        :np.array(test_preds).ravel(),
    #         'test_preds_ir'     :predictions_ir}
    # df_test=pd.DataFrame(d_test)
    # df_test.to_csv("test_IR.csv")

    #score_test_ir=ir.score(test_preds,y_test)
    score_test_ir=0

    return predictions_ir,y_ir_pred,ir.get_params(deep=True),score_test_ir

def SGDRegressor_grid_pred(X_train, X_test, y_train_normalized, y_train_mean):
    """ Note: when you use this method, you should test it on a separate test set!
    Squared loss is giving problems
    """
    tuned_parameters = [{'loss' : ['squared_loss'],
                         # 'warm_start' : ['False'],
                         # 'alpha': [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1], # Causes memory error when included
                         # 'eta0': [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1], # Causes memory error when included
                         # 'l1_ratio' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], # Causes memory error when included
                         'n_iter' : [150]},
                         {'loss' : ['epsilon_insensitive'],
                          'n_iter' : [150],
                          'epsilon' : [0,0.0001,0.001,0.01],
                          'power_t' : [0.1,0.3,0.5],
                          'l1_ratio' : [0,0.15,0.5,1.0],
                          'eta0': [0.0001,0.001,0.01,0.1],
                          'alpha': [0.0001,0.001,0.01,0.1]
                          }]
                         # 'power_t' : [0.1,0.2,0.25,0.3,0.4,0.5]}]

    # Consider another score_func?
    clf = GridSearchCV(SGDRegressor(fit_intercept=False, shuffle=True,verbose=0,penalty='elasticnet'), tuned_parameters, score_func=mean_absolute_error, n_jobs=-1)
    # clf = GridSearchCV(SGDRegressor(fit_intercept=False, shuffle=True,verbose=1,penalty='elasticnet'), tuned_parameters, score_func=mean_absolute_error)
    clf = clf.fit(X_train,y_train_normalized.ravel())
    predictions = clf.predict(X_test) + y_train_mean

    print "Grid search metrics"    
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)

def Append_2_log(timestamp,total_time,train_size,test_size,threshold,AUC,AUC_IR,RIG,RIG_IR,selected_list,note="null"):
    write_str="\n"+str(timestamp)+","+str(total_time)+","+str(train_size)+","+str(test_size)+","+str(threshold)+","+str(AUC)+","+str(AUC_IR)+","+str(RIG)+","+str(RIG_IR)+","+str(selected_list).replace(",","|")+","+note
    with open("log.csv", "a") as myfile:
        myfile.write(write_str)


if __name__=="__main__":
    main(sys.argv[1:])











