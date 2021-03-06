from mpl_toolkits.mplot3d import Axes3D #For Basic ploting
from sklearn.preprocessing import StandardScaler #Preprocessing
from sklearn import preprocessing    # Preprocessing
from sklearn.naive_bayes import GaussianNB #import gaussian naive bayes model
from sklearn.tree import DecisionTreeClassifier #import Decision tree classifier
from sklearn import metrics  #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from yellowbrick.classifier import ClassificationReport
from sklearn.naive_bayes import GaussianNB
from yellowbrick.features import PCA
from sklearn import preprocessing        
nRowsRead = None # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

# (x,y)
train_data = pd.read_csv('UNSW_NB15_training-set.csv', delimiter='\t', nrows = nRowsRead)
train_data.dataframeName = 'UNSW_NB15_training-set.csv'
nRow, nCol = train_data.shape
print(f'There are {nRow} rows and {nCol} columns')        
nRowsRead = None # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
test_data = pd.read_csv('UNSW_NB15_testing-set.csv', delimiter='\t', nrows = nRowsRead)
test_data.dataframeName = 'UNSW_NB15_testing-set.csv'
nRow, nCol = test_data.shape
print(f'There are {nRow} rows and {nCol} columns')        
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

#train_data['label'] = label_encoder.fit_transform(train_data['label'])
#test_data['label'] = label_encoder.fit_transform(test_data['label'])    

#
#train_data['srcip'] = label_encoder.fit_transform(train_data['srcip'])
#test_data['srcip'] = label_encoder.fit_transform(test_data['srcip'])    
#
#
#train_data['dstip'] = label_encoder.fit_transform(train_data['dstip'])
#test_data['dstip'] = label_encoder.fit_transform(test_data['dstip'])    
print(train_data.head())

train_data['proto'] = label_encoder.fit_transform(train_data['proto'])
test_data['proto'] = label_encoder.fit_transform(test_data['proto'])    


train_data['state'] = label_encoder.fit_transform(train_data['state'])
test_data['state'] = label_encoder.fit_transform(test_data['state'])    


train_data['service'] = label_encoder.fit_transform(train_data['service'])
test_data['service'] = label_encoder.fit_transform(test_data['service'])    


names = train_data.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(train_data)
train_data = pd.DataFrame(train_data, columns=names)

X_train = train_data.drop('label',axis=1)
X_test = test_data.drop('label',axis=1)
y_train = train_data['label']
y_test = test_data['label']



# create gaussian naive bayes classifier

classes=["BENIGN","Attack"]


import time

#visualizer = PCA(scale=True, classes=classes)
#visualizer.fit_transform(X_train, y_train)
#visualizer.show()
gnb= GaussianNB()
#Train the model using the training sets
start = time.time()
gnb.fit(X_train,y_train)
end = time.time()
print("Training time : ",end-start)
#Predict the response for test dataset
start_testing= time.time()
predicted = gnb.predict(X_test)
end_testing= time.time()
print("Testing Time : " , end_testing - start_testing)
# Model Accuracy, how often is the classifier correct?




def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN = (perf_measure(y_test,predicted))

y_pred = predicted
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
# Fmeasure = (2 ∗ Precision ∗ Recall)∕(Precision+ Recall)

print("ACC : ",ACC)
Sensitivity = (TP)/(TP + FN)
print("Sensitivity : ", Sensitivity)
Specificity = (TN)/(TN + FP)
print("Specificity : ", Specificity)


#
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, predicted, labels=[0,1])
#
#
#
#
#print(cm)
#import seaborn
#import matplotlib.pyplot as plt
# 
# 
#def plot_confusion_matrix(data, labels, output_filename):
#    """Plot confusion matrix using heatmap.
# 
#    Args:
#        data (list of list): List of lists with confusion matrix data.
#        labels (list): Labels which will be plotted across x and y axis.
#        output_filename (str): Path to output file.
# 
#    """
#    seaborn.set(color_codes=True)
#    plt.figure(1, figsize=(9, 6))
# 
#    plt.title("Confusion Matrix")
# 
#    seaborn.set(font_scale=1.4)
#    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
# 
#    ax.set_xticklabels(labels)
#    ax.set_yticklabels(labels)
# 
#    ax.set(ylabel="True Label", xlabel="Predicted Label")
# 
#    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
#    plt.close()
# 
## create confusion matrix
#plot_confusion_matrix(cm, ["BENIGN","DoS Hulk" , "DoS slowloris"], "confusion_matrix_GNB.png")
#
#
#"""
#
#Precision quantifies the number of positive class predictions that actually belong to the positive class.
#
#Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.
#
#F-Measure provides a single score that balances both the concerns of precision and recall in one number
#"""
#
## Precision = TruePositives / (TruePositives + FalsePositives)

precision = metrics.precision_score(y_test, y_pred, average='micro')
print('Precision: %.3f' % precision)


# Recall = TruePositives / (TruePositives + FalseNegatives)

recall = metrics.recall_score(y_test, y_pred, average='micro')
print('Recall: %.3f' % recall)

# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
score = metrics.f1_score(y_test, y_pred, average='micro')
print('F-Measure: %.3f' % score)