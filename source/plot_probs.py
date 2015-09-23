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


nb_site_file="/Users/tan/dse/test_preds_NB_site_1000_1381492647.02.csv"
nb_all_file="/Users/tan/dse/test_preds_NB_all_1000_1381489376.2.csv"


threshold = 1000
data ="DATA = Train:8M Test:2M - Sampled September 2013\n\n"
model = "MODEL = MultinomialNB(alpha=0.1, fit_prior=True), threshold = "+str(threshold)+"\n"
features_all="[ad-account-advertiser_account-app_id-campaign-country-device-hour-rtb-site-weekday]"
features_site="[site]"

#======
#======
#=======
test_df=pd.read_csv(nb_site_file)
predictions     = test_df["Predictions"]
#predictions_ir  = test_df["predSIR"]
y_test          = test_df["y_test"]

prob_target_step = 10000

predictions_mean=np.ones(len(predictions))*predictions.mean()
#predictions_ir_mean=np.ones(len(predictions))*predictions_ir.mean()
y_test_mean=np.ones(len(predictions))*y_test.mean()



ax = plt.subplot(2,1,1)

#ax.annotate(features_site, xy=(round(len(y_test)/2), (max(predictions)/6)*5),color='k',size=25)

sorted_y_test=np.array(y_test)[np.array(predictions).argsort()]
prob_target=GetTargetProb(sorted_y_test,prob_target_step)
ax.plot(prob_target,'g',label='Target Values',linewidth=1)
ax.plot(y_test_mean,'g--',label='Mean Target Value',linewidth=0.5)
ax.annotate(str(y_test.mean()), xy=(round(len(y_test)/5), y_test.mean()),color='g')


ax.plot(np.sort(predictions),label='Predictions',linewidth=2)
ax.plot(predictions_mean,'b--', label='Mean Predictions',linewidth=0.5)
ax.annotate(str(predictions.mean()), xy=(round(len(y_test)/20), predictions.mean()),color='b')
ax.annotate("max_pred = "+str(max(predictions)), xy=(round(len(y_test)/20), 0.01),color='b')
ax.annotate("", xy=(round(len(y_test)/20), 0.01),xytext=(round(len(y_test)/20), predictions.mean()),color='b',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

#ax.plot(np.array(predictions_ir)[np.array(predictions).argsort()],'r',label='Predictions (Iso. Reg.)',linewidth=2)
#ax.plot(predictions_ir_mean,'r--',label='Mean Predictions (Iso. Reg.)',linewidth=0.5)
#ax.annotate(str(predictions_ir.mean()), xy=(round(len(y_test)/10), predictions_ir.mean()),color='r')

#ax.legend(handles[::-1], labels[::-1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper left')

plt.ylabel("Probabilities (All probabilities sorted by [predictions])")
plt.xlabel("Impression")


title_str=model+data+features_site
plt.title(title_str)


#=============
#============
#=============





test_df=pd.read_csv(nb_all_file)
predictions     = test_df["Predictions"]
#predictions_ir  = test_df["predSIR"]
y_test          = test_df["y_test"]

prob_target_step = 10000

predictions_mean=np.ones(len(predictions))*predictions.mean()
#predictions_ir_mean=np.ones(len(predictions))*predictions_ir.mean()
y_test_mean=np.ones(len(predictions))*y_test.mean()



ax = plt.subplot(2,1,2)



sorted_y_test=np.array(y_test)[np.array(predictions).argsort()]
prob_target=GetTargetProb(sorted_y_test,prob_target_step)
ax.plot(prob_target,'g',label='Target Values',linewidth=1)
ax.plot(y_test_mean,'g--',label='Mean Target Value',linewidth=0.5)
ax.annotate(str(y_test.mean()), xy=(round(len(y_test)/5), y_test.mean()),color='g')


ax.plot(np.sort(predictions),label='Predictions',linewidth=2)
ax.plot(predictions_mean,'b--', label='Mean Predictions',linewidth=0.5)
ax.annotate(features_all, xy=(round(len(y_test)/18), (max(predictions)/6)*5),color='k',size=10)
ax.annotate(str(predictions.mean()), xy=(round(len(y_test)/20), predictions.mean()),color='b')
ax.annotate("max_pred = "+str(max(predictions)), xy=(round(len(y_test)/20), 0.02),color='b')
ax.annotate("", xy=(round(len(y_test)/20), 0.02),xytext=(round(len(y_test)/20), predictions.mean()),color='b',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

#ax.plot(np.array(predictions_ir)[np.array(predictions).argsort()],'r',label='Predictions (Iso. Reg.)',linewidth=2)
#ax.plot(predictions_ir_mean,'r--',label='Mean Predictions (Iso. Reg.)',linewidth=0.5)
#ax.annotate(str(predictions_ir.mean()), xy=(round(len(y_test)/10), predictions_ir.mean()),color='r')

#ax.legend(handles[::-1], labels[::-1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper left')

plt.title(features_all)

plt.ylim(0,0.035)

plt.ylabel("Probabilities (All probabilities sorted by [predictions])")
plt.xlabel("Impression")





















#ax.show()