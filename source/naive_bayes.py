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
from sklearn.naive_bayes        import MultinomialNB
from sklearn.externals          import joblib

############### READ TRAINING FILE ###############
def main(argv):
    args             = parse_args()
    input_train_file = args.input_train_file
    input_test_file  = args.input_test_file
    output_file      = args.output_file
    delim            = args.delim 
    threshold        = 3000

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
    print "entropy2:   ", entropy2
    log_loss_preds2 = np.zeros(shape=(len(y_test),2))
    for i in range(0,len(y_test)-1):
        log_loss_preds2[i] = [1-y_test_mean,y_test_mean]
    
    sum1 = 0

    # for i in range(0,len(y_test)-1):
    # for i,truth in enumerate(y_test):
        # sum1 += float(truth)*math.log(log_loss_preds2[i][1])+(1-float(truth))*math.log(log_loss_preds2[i][0])

    # cross_entropy3 = sum1/len(y_test)
    # rig_test3        = 1 + cross_entropy3/entropy2

    # cross_entropy2   = -log_loss(y_test.astype(int).ravel(), log_loss_preds2)
    # rig_test2        = 1 + cross_entropy2/entropy2

    # for i in range(0,len(y_test)-1):
    #     -(y_test_mean*math.log(y_test_mean)+(1-y_test_mean)*math.log(1-y_test_mean))

    # print "Cross-entropy2: ", cross_entropy2
    # print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test2

    # print "Cross-entropy3: ", cross_entropy3
    # print "This is the RIG for the GTA (should be 0 or close to 0): ", rig_test3

    # Some quick testing ##########################

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

    auc_log = []
    rig_log = []
    mean_pred_log = []
    max_pred_log = []

    # If an attribute has only 1 value or all missing (auc=0.5), then remove it from the model / selection process
    # If an attrbitute deteriorates performance, shall we simply remove it from the full list?
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
                predictions     = MultinomialNB_pred(X_train, X_test, y_train)
                auc             = auc_score(y_test.astype(int).ravel(), predictions)

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
                auc             = 0.5
                rig             = 0 # rig_model

            # # keep the attribute with the highest auc score; e.g. add it to model
            # # rewrite into if, elid, else
            # # AUC selection criterion
            # if auc > auc_model:
            #     auc_model = auc
            #     rig_model = rig
            #     best_attribute = attribute
            # # Tie breaking
            # # if auc == auc_model:
            # #     if random.random() > 0.5:
            # #         auc_model = auc
            # #         rig_model = rig
            # #         best_attribute = attribute
            # # If we leave this out, the full space is explored => marginally better performance
            # if auc < auc_model_old:
            #     full_list.remove(attribute)


            # AUC selection criterion alternative #1: AUC selection but RIG needs to be positive
            if auc > auc_model:
                if rig > 0:
                    auc_model = auc
                    rig_model = rig
                    best_attribute = attribute
            # Tie breaking
            # if auc == auc_model:
            #     if random.random() > 0.5:
            #         auc_model = auc
            #         rig_model = rig
            #         best_attribute = attribute
            # If we leave this out, the full space is explored => marginally better performance
            if auc < auc_model_old:
                full_list.remove(attribute)

            # # Relative Information Gain selection criterion
            # if rig > rig_model:
            #     if auc != auc_model: # if the auc == auc_model, then actually rig == rig_model, so no improvement
            #         auc_model = auc
            #         rig_model = rig
            #         best_attribute = attribute

            # # If we leave this out, the full space is explored => marginally better performance
            # if rig <= rig_model_old:
            #     full_list.remove(attribute)



            selected_list.remove(attribute)
            print "auc:                ", auc
            print "rig:                ", rig
            print "cross entropy:      ", cross_entropy
            print "average prediction: ", np.mean(predictions)
            print "min prediction:     ", np.min(predictions)
            print "max prediction:     ", np.max(predictions)
            print "GTA train:          ", y_train_mean
            print "GTA test:           ", y_test_mean

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
    print "threshold:           ", threshold
    print "AUC:                 ", auc_model
    print "RIG:                 ", rig_model
    print "AUC log:             ", auc_log
    print "RIG log:             ", rig_log
    print "Mean prediction log: ", mean_pred_log
    print "Max prediction log:  ", max_pred_log


    ############### SPLIT DATA FROM TARGETS ###############

    # y_train, train = split_target(train)
    # y_test, test   = split_target(test)

    # Should we use the built-in normalization functions of Scikit-learn?
    # http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
    # from sklearn import preprocessing
    # X_train_scaled = preprocessing.scale(X_train)

    # Question: how to transform back predictions to the proper scale again?
    # Use StandardScaler.inverse_transform() or sth?
    # y_train_mean = np.mean(y_train)
    # y_train_normalized = y_train - y_train_mean

    # scaler = StandardScaler().fit(y_train)
    # y_train_normalized2 = scaler.transform(y_train)

    ########################## DictVect ##########################

    # NB Potentially this transformation can be made more efficient with
    # FeatureHasher from sklearn.feature_extraction. This transorms the data into
    # a SciPy sparse matrix. However, the regressor functions take NumPy arrays
    # as arguments. The problem is that I get a segmentation fault when trying
    # to transform the SciPy sparse matrix back to NumPy with todense(). This
    # is only a problem with matrix that have a certain size and up.
    #
    # Furthermore, the fitting of the model with a converted matrix (SciPy -> NumPy)
    # takes a *much* longer time than a non-converted one. Finally, note that the
    # regressor methods can accept SciPy sparse matrices, but the output is not right.

    # X_train, X_test = vectorize_dicts(train,test)

    ########################## ALTERNATIVE: FeatHash ##########################

    # The alternative: feature hashing
    # print "Feature Hasher:"
    # print "Hashing...."
    # t10 = time()
    # feat = FeatureHasher(input_type="string")
    # X_train = feat.transform(train).todense()
    # print "Succesful with train"
    # X_test  = feat.transform(test).todense()
    # print "done in %fmin" % ((time() - t10)/60)
    # print "n_samples: %d, n_features: %d" % X_train.shape

    ############### SETTING THE MODEL ###############

    # predictions = MultinomialNB_pred(X_train, X_test, y_train)

    # predictions = SGDRegressor_grid_pred(X_train, X_test, y_train_normalized, y_train_mean)
    # predictions = SGDRegressor_grid_pred(X_train, X_test, y_train_normalized2, y_train_mean)

    # clf = SGDRegressor(alpha=0.0001, eta0=0.001, n_iter=10, fit_intercept=False, shuffle=True,verbose=1)

    # print "Fitting the model...."
    # t1 = time()
    # clf = clf.fit(X_train,y_train_normalized)
    # print "done in %fmin" % ((time() - t1)/60)

    # print "Assigning predictions to test set...."
    # t2 = time()
    # predictions = clf.predict(X_test)
    # predictions += y_train_mean # Adding the mean back in

    # print "done in %fmin" % ((time() - t2)/60)

    # print "Some testing:"
    # print "The original training matrix"
    # print "n_samples: %d, n_features: %d" % X_train.shape
    # print "Example:"
    # pprint(X_train[1])
    # pprint(X_train[1].todense())
    # pprint(np.sum(X_train[1].todense()))

    ############### SCORE ###############

    # auc_score1 = auc_score(y_test.astype(int).ravel(), predictions)
    # print "The AUC score: ", auc_score1
    # mae = mean_absolute_error(y_test.astype(int).ravel(), predictions)
    # print "Mean absolute error: ", mae

    ############### SAVE THE PREDICTIONS ###############

    # predictions_to_file(predictions,input_test_file,output_file,delim)

    ############### SAVE THE MODEL ###############

    # To do: put this as a command line argument?
    # Something is going wrong here when we do a grid search, so commenting it out
    # joblib.dump(clf, 'bias_model.pkl') 

    ######################################

    print "done with EVERYTHING in %fmin" % ((time() - t100)/60)

# Give the user extra command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Bias model for CTR estimation")

    parser.add_argument("input_train_file",     type=str, help="the input train file")
    parser.add_argument("input_test_file",      type=str, help="the input test file")
    parser.add_argument("output_file",          type=str, help="the output file with predictions")
    parser.add_argument("--delim", default=',', type=str, help="the delimiter in the train, test, and predictions file")

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

    # can we not simply output dicts directly?
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

def MultinomialNB_pred(X_train, X_test, y_train):
    clf = MultinomialNB(alpha=0.1, fit_prior=True)
    clf = clf.fit(X_train,y_train)
    predictions = clf.predict_proba(X_test) # these are predictions for both classes, so non-clicks and clicks
    
    # Get only the predictions for clicks
    predictions_click = []
    for pred in predictions:
        predictions_click.append(pred[1])
    return predictions_click

if __name__=="__main__":
    main(sys.argv[1:])