import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
import time
data = pd.read_csv("UNSW-NB15_2_2_2.csv", encoding='utf8')
print(data.head())
cls = list(data["Label"])
predict = "Label"
X = list(zip(data["srcip"].tolist(), data['sport'].tolist(), data["dstip"].tolist(), data['dsport'].tolist(),
             data["proto"].tolist(), data["state"].tolist(), data['dur'].tolist(), data['sbytes'].tolist(),
             data['dbytes'].tolist(), data['sttl'].tolist(), data['dttl'].tolist(), data['sloss'].tolist(),
             data['sloss'].tolist(), data["service"].tolist(), data['sload'].tolist(), data['dload'].tolist(),
             data['spkts'].tolist(), data['swin'].tolist(), data['dwin'].tolist(), data['stcpb'].tolist(),
             data['dtcpb'].tolist(), data['smeansz'].tolist(), data['dmeansz'].tolist(), data['trans_depth'].tolist(),
             data['res_bdy_len'].tolist(), data['sjit'].tolist(), data['djit'].tolist(), data['stime'].tolist(),
             data['ltime'].tolist(), data['sintpkt'].tolist(), data['dintpkt'].tolist(), data['tcprtt'].tolist(),
             data['synack'].tolist(), data['ackdat'].tolist(), data['is_sm_ips_ports'].tolist(),
             data['ct_state_ttl'].tolist(), data['ct_flw_http_mthd'].tolist(), data['is_ftp_login'].tolist(),
             data['ct_ftp_cmd'].tolist(), data['ct_srv_src'].tolist(), data['ct_srv_dst'].tolist(),
             data['ct_dst_ltm'].tolist(), data['ct_src_ltm'].tolist(), data['ct_src_dport_ltm'].tolist(),
             data['ct_dst_sport_ltm'].tolist(), data['ct_dst_src_ltm'].tolist()))

y = list(cls)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100
}
start = time.time()
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
model = XGBClassifier(**params).fit(X_train,y_train)
end = time.time()
print("Training time : ",end-start)
# make predictions on test data
start_testing= time.time()
y_pred = model.predict(X_test)
end_testing= time.time()
print("Testing Time : " , end_testing - start_testing)
# Model Accuracy
def perf_measure(y_test, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_test[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_test[i] != y_pred[i]:
            FP += 1
        if y_test[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_test[i] != y_pred[i]:
            FN += 1
   # print(TP, FP, TN, FN)

    return (TP, FP, TN, FN)


TP, FP, TN, FN = (perf_measure(y, y_pred))

ACC = (TP+TN)/(TP+FP+FN+TN)
#Sensitivity = (TP)/(TP + FN)
#print("Sensitivity : ", Sensitivity)
#Specificity = (TN)/(TN + FP)
#print("Specificity : ", Specificity)

print("ACC : ", ACC)
precision = metrics.precision_score(y_test, y_pred, average='micro')
print('Precision: %.3f' % precision)


# Recall = TruePositives / (TruePositives + FalseNegatives)

recall = metrics.recall_score(y_test, y_pred, average='micro')
print('Recall: %.3f' % recall)

# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
score = metrics.f1_score(y_test, y_pred, average='micro')
print('F-Measure: %.3f' % score)