from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn


data = pd.read_csv('./Data/wine.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

ndata = data.shape[0]
ncolumn = data.shape[1]

train_rate = 0.7
ntrain = int(ndata * train_rate)
train_index = range(ntrain)
test_index = range(ntrain, ndata)

train, test = data.iloc[train_index,], data.iloc[test_index,]
train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]

train_accuracy = []
test_accuracy = []

train_precision = []
test_precision = []

train_recall = []
test_recall = []

train_specificity= []
test_specificity= []

train_f1measure= []
test_f1measure= []

train_gmean= []
test_gmean = []

for i in range(1,100):

    regr = RandomForestClassifier(max_depth=i, random_state=0, n_estimators=100)
    regr.fit(train_x, train_y.values.ravel())

    estimate = regr.predict(train_x)
    C1 = confusion_matrix(train_y, estimate)
    TN, FP, FN, TP = C1.ravel()

    Accuracy = accuracy_score(train_y, estimate)
    Precision = float(TP / (TP + FP))
    Recall = float(TP / (TP + FN))
    Specificity = float(TN / (TN + FP))
    F1measure = float(2 * Precision * Recall / (Precision + Recall))
    Gmean = float(np.sqrt(Precision * Recall))

    estimates2 = regr.predict(test_x)
    C2 = confusion_matrix(test_y, estimates2)
    TN2, FP2, FN2, TP2 = C2.ravel()

    Accuracy2 = accuracy_score(test_y, estimates2)
    Precision2 = float(TP2 / (TP2 + FP2))
    Recall2 = float(TP2 / (TP2 + FN2))
    Specificity2 = float(TN2 / (TN2 + FP2))
    F1measure2 = float(2 * Precision2 * Recall2 / (Precision2 + Recall2))
    Gmean2 = float(np.sqrt(Precision2 * Recall2))

    train_gmean.append(Gmean)
    test_gmean.append(Gmean2)


fig, ax = plt.subplots()
ax.plot(range(1,100),train_gmean, color ='black', marker='o', label = 'G-mean of train')
ax.plot(range(1,100),test_gmean, color ='blue', marker='x', label = 'G-mean of test')
plt.legend()
plt.xlabel('Depth of the tree')
plt.ylabel('G-mean')
ax.set(xticks=range(0,100,10),xlim=[0,101],ylim=[0.5,1])
plt.tight_layout()

max_depth = np.argmax(test_gmean)+1
regr = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=100)
regr.fit(train_x, train_y.values.ravel())

estimates = regr.predict(train_x)
C=confusion_matrix(train_y,estimates)
TN, FP, FN, TP = C.ravel()

Accuracy= accuracy_score(train_y,estimates)
Precision=float(TP/(TP+FP))
Recall=float(TP/(TP+FN))
Specificity=float(TN/(TN+FP))
F1measure=float(2*Precision*Recall/(Precision+Recall))
Gmean=float(np.sqrt(Precision*Recall))

print('***max_depth is %d***'%(max_depth))
print("This solution is computed using train data")
print(C)
print("Accuracy using train data is: %.3f"%(Accuracy))
print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision, Recall, Specificity, F1measure, Gmean))
print("Type 1 error : %.3f, Type 2 error : %.3f\n"%(1-Specificity, 1-Recall))

estimates2 = regr.predict(test_x)

C2=confusion_matrix(test_y,estimates2)
TN2, FP2, FN2, TP2 = C2.ravel()

Accuracy2 = accuracy_score(test_y, estimates2)
Precision2 = float(TP2 / (TP2 + FP2))
Recall2 = float(TP2 / (TP2 + FN2))
Specificity2 = float(TN2 / (TN2 + FP2))
F1measure2 = float(2 * Precision2 * Recall2 / (Precision2 + Recall2))
Gmean2 = float(np.sqrt(Precision2 * Recall2))

print('***max_depth is %d***' % (max_depth))
print("This solution is computed using test data")
print(C2)
print("Accuracy using test data is: %.3f" % (Accuracy2))
print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" % (
Precision2, Recall2, Specificity2, F1measure2, Gmean2))
print("Type 1 error : %.3f, Type 2 error : %.3f\n" % (1 - Specificity2, 1 - Recall2))

df_cm = pd.DataFrame(C, ['Actual N','Actual P'],['Predicted N','Predicted P'])
df_cm2 = pd.DataFrame(C2, ['Actual N','Actual P'],['Predicted N','Predicted P'])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set(title='Confusion Matrix of Train Data')

ax2 = fig.add_subplot(212)
ax2.set(title='Confusion Matrix of Test Data')

sn.heatmap(df_cm, annot=True, fmt='d', ax=ax1, annot_kws={"size": 16})
sn.heatmap(df_cm2, annot=True, fmt='d', ax=ax2, annot_kws={"size": 16})
plt.tight_layout()
plt.show()
