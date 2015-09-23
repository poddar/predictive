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
from   scipy import sparse
from   time   import time
from   pprint import pprint
from   copy   import copy

from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.metrics            import auc_score, mean_absolute_error, log_loss
from sklearn.externals          import joblib
from sklearn.linear_model       import LogisticRegression

############### READ TRAINING FILE ###############
def main(argv):
    args             = parse_args()
    input_train_file = args.input_train_file
    input_test_file  = args.input_test_file
    output_file      = args.output_file
    delim            = args.delim 
    threshold        = 0

    t100 = time()
    print "Starting!"
    print "Threshold: ", threshold

    ############### READ TEST AND TRAIN FILES ###############

    train     = file_to_dict(input_train_file,delim)
    test      = file_to_dict(input_test_file,delim)

    # 1. Initialize model = []
    # 2.a Read header into an array: "search_space"
    # 2.b Delete out 'clicks'
    # 3. Iterate through search_space, construct training data: ["model",'clicks']

    # Perhaps rename train & test earlier to note the diff better
    y_train, train     = split_target(train)
    y_test, test       = split_target(test)
    y_train_mean       = np.mean(y_train)
    y_test_mean        = np.mean(y_test) # Not used, but just to check
    y_train_normalized = y_train - y_train_mean

    entropy = -(y_train_mean*math.log(y_train_mean)+(1-y_train_mean)*math.log(1-y_train_mean))
    print "entropy:   ", entropy
    print "GTA train: ", y_train_mean
    print "GTA test:  ", y_test_mean
    auc_model = 0.5

    # Some quick testing ##########################
    log_loss_preds = np.zeros(shape=(len(y_test),2))
    for i in range(0,len(y_test)-1):
        log_loss_preds[i] = [1-y_train_mean,y_train_mean]
    cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
    rig_test        = 1 + cross_entropy/entropy

    print "Cross-entropy: ", cross_entropy
    print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test

    # NB This is the correct entropy
    entropy2 = -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean))
    sum1 = 0

     for i in range(0,len(y_test)-1):
         sum1 += log_loss_preds2[i][1]*math.log(log_loss_preds2[i][1])+log_loss_preds2[i][0]*math.log(log_loss_preds2[i][0])

     cross_entropy3 = sum1/len(y_test)
     rig_test3        = 1 + cross_entropy3/entropy2

     cross_entropy2   = -log_loss(y_test.astype(int).ravel(), log_loss_preds2)
     rig_test2        = 1 + cross_entropy2/entropy2



    print "Cross-entropy2: ", cross_entropy2
    print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test2

    print "Cross-entropy3: ", cross_entropy3
    print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test3

    # Some quick testing

    # relative information gain
    # NB rig will never be zero due to epsilon clipping.
    # Be careful when deciding whether to keep/discard variables on the basis of having a marginally higher rig!
    # auc: 0.5
    # rig: 0.000962618510464
    rig_model = 0

    full_list = header(input_train_file,delim)
    selected_list   = []
    complement      = header(input_train_file,delim)

    len_full_list = len(full_list)

    # If an attribute has only 1 value or all missing (auc=0.5), then remove it from the model / selection process
    # If an attrbitute deteriorates performance, shall we simply remove it from the full list?
    time_stamp=time()
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

            print selected_list

            new_train = []
            new_test  = []

            for row in train:
                new_train.append(dict((k, row[k]) for k in selected_list))

            for row in test:
                new_test.append(dict((k, row[k]) for k in selected_list))

            X_train, X_test = vectorize_dicts(new_train,new_test,threshold)
            if X_train.getnnz() > 0:
                predictions     = LogisticRegression_pred(X_train, X_test, y_train)
                auc             = auc_score(y_test.astype(int).ravel(), predictions)

                # Can be done prettier
                log_loss_preds = np.zeros(shape=(len(predictions),2))
                for i,pred in enumerate(predictions):
                    log_loss_preds[i] = [1-pred,pred]

                 print "type(log_loss_preds): ", type(log_loss_preds)
                 print "log_loss_preds.shape: ", log_loss_preds.shape

                # Warning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
                # Ignore the warning
                cross_entropy   = -log_loss(y_test.astype(int).ravel(), log_loss_preds)
                rig             = 1 + cross_entropy/entropy2
            else:
                predictions     = np.zeros(1) # hack
                auc             = 0.5
                rig             = 0 # rig_model

            # keep the attribute with the highest auc score; e.g. add it to model

            # # Relative Information Gain selection criterion
            if rig > rig_model:
                if auc != auc_model: # if the auc == auc_model, then actually rig == rig_model, so no improvement
                    auc_model = auc
                    rig_model = rig
                    best_attribute = attribute

            # If we leave this out, the full space is explored => marginally better performance
            if rig <= rig_model_old:
                full_list.remove(attribute)

            
            print "auc:                ", auc
            print "rig:                ", rig
            print "cross entropy:      ", cross_entropy
            print "average prediction: ", np.mean(predictions)
            print "min prediction:     ", np.min(predictions)
            print "max prediction:     ", np.max(predictions)
            print "GTA:                ", y_train_mean

            results_str=str(selected_list).replace(",","|")+","+str(auc)+","+str(rig)+"\n"
            file_name="LR_resultes_log_"+str(time_stamp)+".csv"
            with open(file_name, "a") as myfile:
                myfile.write(results_str)

            selected_list.remove(attribute)

        if len(best_attribute) > 0:
            selected_list.append(best_attribute)

        print "Best selected_list so far: ", selected_list
        print "auc:                       ", auc_model
        print "rig:                       ", rig_model

        if len_selected_list == len(selected_list):
            break

    print "best model: ", selected_list
    print "AUC:        ", auc_model
    print "RIG:        ", rig_model


print "done with EVERYTHING in %fmin" % ((time() - t100)/60)

# Give the user extra command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Bias model for CTR estimation")

    parser.add_argument("input_train_file",     type=str, help="the input train file")
    parser.add_argument("input_test_file",      type=str, help="the input test file")
    parser.add_argument("output_file",          type=str, help="the output file with predictions")
    parser.add_argument("--delim", default=',', type=str, help="the delimiter in the train, test, and predictions file")

    args = parser.parse_args()
    return args

def predictions_to_file(predictions,input_test_file,output_file,delim):
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

    if len(selected_features) > 0:
        X_train_selected = X_train[:,selected_features]
        X_test_selected  = X_test[:,selected_features]
    else:
        X_train_selected = sparse.csr_matrix([0]) # Hack to get zero-length matrices
        X_test_selected  = sparse.csr_matrix([0])

    return X_train_selected, X_test_selected

def file_to_dict(filename,delim):
    """ Method for reading in train or test data
    """
    dict_array = []

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
    
    return y, array_of_dicts

def LogisticRegression_pred(X_train, X_test, y_train):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None) # defaults
    clf = clf.fit(X_train,y_train)
    predictions = clf.predict_proba(X_test) # these are predictions for both classes, so non-clicks and clicks
    
    # Get only the predictions for clicks - is this necessary for LR?
    predictions_click = []
    for pred in predictions:
        predictions_click.append(pred[1])
    return predictions_click

if __name__=="__main__":
    main(sys.argv[1:])