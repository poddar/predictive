import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc_score, mean_absolute_error,log_loss,mean_squared_error

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



test_df_all=pd.read_csv("/Users/tan/dse/test_preds_NB_all_1000_1381489376.2.csv")
test_df_site=pd.read_csv("/Users/tan/dse/test_preds_NB_site_1000_1381492647.02.csv")

hist_step=1000

threshold = 1000
data ="DATA = Train:8M Test:2M - Sampled September 2013\n\n"
model = "MODEL = MultinomialNB(alpha=0.1, fit_prior=True), threshold = "+str(threshold)+"\n"
features_all="[ad-account-advertiser_account-app_id-campaign-country-device-hour-rtb-site-weekday]"
features_site="[site]"


predictions     = test_df_all["Predictions"]
#predictions_ir  = test_df_all["predSIR"]
y_test          = test_df_all["y_test"]

predictions_site    = test_df_site["Predictions"]
#predictions_ir_site = test_df_site["predSIR"]
y_test_site         = test_df_site["y_test"]

plt.figure(1)



#====
#====
#====




ax = plt.subplot(2,1,1)


title_str=model+data+features_site
plt.title(title_str)
weights = np.ones_like(predictions_site)/len(predictions_site)
n_site, bins_site, patches_site = ax.hist(predictions_site, hist_step, histtype='bar',weights=weights)
plt.xlim(0,0.05)
#ax.set_yscale('log')


#ax = plt.subplot(2,2,4)
#weights_ir_site = np.ones_like(predictions_ir_site)/len(predictions_ir_site)
#n_ir_site, bins_ir_site, patches_ir_site = ax.hist(predictions_ir_site, 100, histtype='bar',weights=weights_ir_site)


plt.ylabel("Distribution")
plt.xlabel("Probabilities")



#====
#====
#====



ax = plt.subplot(2,1,2)
plt.title(features_all)

weights = np.ones_like(predictions)/len(predictions)
n, bins, patches = ax.hist(predictions, hist_step, histtype='bar',weights=weights)
plt.xlim(0,0.05)
#ax.set_xscale('log')





#ax = plt.subplot(2,2,3)
#weights_ir = np.ones_like(predictions_ir)/len(predictions_ir)
#n_ir, bins_ir, patches_ir = ax.hist(predictions_ir, 100, histtype='bar',weights=weights_ir)
plt.ylabel("Distribution")
plt.xlabel("Probabilities")






#======= RATIO =====#
ratio_arr = predictions/predictions_site
plt.figure(2)
ax = plt.subplot(1,1,1)
ratio_str="predictions_all / predictions_site"
title_str=model+data+ratio_str
plt.title(title_str)
weights = np.ones_like(ratio_arr)/len(ratio_arr)
n, bins, patches = ax.hist(ratio_arr, hist_step, histtype='bar',weights=weights)
plt.xlim(0,10)
#ax.set_yscale('log',basey=2)
plt.xlabel("Ratio")
plt.ylabel("Distribution")




#predictions_mean=np.ones(len(predictions))*predictions.mean()
#predictions_ir_mean=np.ones(len(predictions))*predictions_ir.mean()
#y_test_mean=np.ones(len(predictions))*y_test.mean()



#ax = plt.subplot(1,1,1)
##p1 = ax.figure(2)
#ax.axvline(0.02)
#ax.plot(np.sort(predictions),label='Predictions',linewidth=2)
#ax.plot(predictions_mean,'b--', label='Mean Predictions',linewidth=0.5)
#ax.annotate(str(predictions.mean()), xy=(round(len(y_test)/10), predictions.mean()),color='b')
#
#ax.plot(np.array(predictions_ir)[np.array(predictions).argsort()],'r',label='Predictions (Iso. Reg.)',linewidth=2)
#ax.plot(predictions_ir_mean,'r--',label='Mean Predictions (Iso. Reg.)',linewidth=0.5)
#ax.annotate(str(predictions_ir.mean()), xy=(round(len(y_test)/10), predictions_ir.mean()),color='r')
#
#sorted_y_test=np.array(y_test)[np.array(predictions).argsort()]
#prob_target=GetTargetProb(sorted_y_test,1000)
#ax.plot(prob_target,'g',label='Target Values',linewidth=1)
#ax.plot(y_test_mean,'g--',label='Mean Target Value',linewidth=0.5)
#ax.annotate(str(y_test.mean()), xy=(round(len(y_test)/10), y_test.mean()),color='g')
#
#
#title_str=model+data+features
#plt.title(title_str)
#
#handles, labels = ax.get_legend_handles_labels()
##ax.legend(handles[::-1], labels[::-1])
#ax.legend(handles, labels,loc='upper left')
#
#plt.ylabel("Probabilities (All probabilities sorted by [predictions])")
#plt.xlabel("Impression")
##ax.show()