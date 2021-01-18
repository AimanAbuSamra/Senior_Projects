import pandas as pd
from sklearn.model_selection import train_test_split
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
# Import Support Vector Classifier
from sklearn.svm import SVC


svc=SVC(probability=True, kernel='linear')
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
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) # 70% training and 30% test
abc =AdaBoostClassifier(n_estimators=3, base_estimator=svc,learning_rate=1)


# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
print("y predict done ...... ")
# Model Accuracy, how often is the classifier correct?
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
        print("XXXXXXXXXXXXX")
   # print(TP, FP, TN, FN)

    return (TP, FP, TN, FN)


TP, FP, TN, FN = (perf_measure(y, y_pred))
TPR = TP / (TP + FN)
# Specificity or true negative rate
TNR = TN / (TN + FP)
# Precision or positive predictive value
try:
    PPV = TP / (TP + FP)
except:
    PPV = 0
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)
# False discovery rate
try:
    FDR = FP / (TP + FP)
except:
    FDR = 0
ACC = (TP+TN)/(TP+FP+FN+TN)
f = open("result.txt", 'a')
f.writelines("TPR : " + str(TPR) + "\n")
f.writelines("TNR : " + str(TNR) + "\n")
f.writelines("PPV : " + str(PPV) + "\n")
f.writelines("NPV : " + str(NPV) + "\n")
f.writelines("FPR : " + str(FPR) + "\n")
f.writelines("FNR : " + str(FNR) + "\n")
f.writelines("FDR : " + str(FDR) + "\n")
f.writelines("ACC : " + str(ACC) + "\n")
f.writelines("TP : " + str(TP) + "\n")
f.writelines("TN : " + str(TN) + "\n")
f.writelines("FN : " + str(FN) + "\n")
f.writelines("FP : " + str(FP) + "\n")

print("TPR : ", TPR)
print("TNR : ", TNR)
print("PPV : ", PPV)
print("NPV : ", NPV)
print("FPR : ", FPR)
print("FNR : ", FNR)
print("FDR : ", FDR)
print("ACC : ", ACC)
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)

try:
    Fmeasure = (2 * PPV * TPR) / (PPV + TPR)
    print("Fmeasure : ", Fmeasure)
    f.writelines("Fmeasure : " + str(Fmeasure) + "\n")
except ZeroDivisionError:
    print("Fmeasure : ", 0)
    f.writelines("Fmeasure : " + str(Fmeasure) + "\n")

# create a for loop of models with different k's


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train score : " + str(train_score))
print("Test score : " + str(test_score))
print("Train Error : " + str(1 - train_score))
print("Test Error : " + str(1 - test_score))
f.writelines("Train score : " + str(train_score) + "\n")
f.writelines("Test score : " + str(test_score) + "\n")
f.writelines("Train Error : " + str(1 - train_score) + "\n")
f.writelines("Test Error : " + str(1 - test_score) + "\n")
f.close()