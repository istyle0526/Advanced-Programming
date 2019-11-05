from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import graphviz

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


for i in range(1,15):
    DT = DecisionTreeClassifier(criterion = 'entropy',max_depth = i, random_state = 1)
    DT.fit(train_x,train_y)

    estimates = DT.predict(train_x)
    C=confusion_matrix(train_y,estimates)
    TN, FP, FN, TP = C.ravel()

    Accuracy= accuracy_score(train_y,estimates)
    Precision=float(TP/(TP+FP))
    Recall=float(TP/(TP+FN))
    Specificity=float(TN/(TN+FP))
    F1measure=float(2*Precision*Recall/(Precision+Recall))
    Gmean=float(np.sqrt(Precision*Recall))

    train_accuracy.append(Accuracy)
    train_precision.append(Precision)
    train_recall.append(Recall)
    train_specificity.append(Specificity)
    train_f1measure.append(F1measure)
    train_gmean.append(Gmean)

    estimates2 = DT.predict(test_x)
    C2=confusion_matrix(test_y,estimates2)
    TN2, FP2, FN2, TP2 = C2.ravel()

    Accuracy2 = accuracy_score(test_y, estimates2)
    Precision2 = float(TP2 / (TP2 + FP2))
    Recall2 = float(TP2 / (TP2 + FN2))
    Specificity2 = float(TN2 / (TN2 + FP2))
    F1measure2 = float(2 * Precision2 * Recall2 / (Precision2 + Recall2))
    Gmean2 = float(np.sqrt(Precision2 * Recall2))

    test_accuracy.append(Accuracy2)
    test_precision.append(Precision2)
    test_recall.append(Recall2)
    test_specificity.append(Specificity2)
    test_f1measure.append(F1measure2)
    test_gmean.append(Gmean2)

fig, ax = plt.subplots()
ax.plot(range(1,15),train_gmean, color ='black', marker='o', label = 'Gmean of train')
ax.plot(range(1,15),test_gmean, color ='blue', marker='x', label = 'Gmean of test')
plt.legend()
plt.xlabel('Depth of the tree')
plt.ylabel('G-mean')
ax.set(xticks=range(1,16,2),xlim=[0,16],ylim=[0,1])

max_depth = np.argmax(test_gmean)+1
DT = DecisionTreeClassifier(criterion = 'entropy',max_depth = max_depth, random_state = 1)
DT.fit(train_x,train_y)
estimates = DT.predict(train_x)

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

estimates2 = DT.predict(test_x)

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

subset_attributes = ['fixed acidity', 'volatile acidity', 'citric acid',
                     'residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
dot_data = tree.export_graphviz(DT, out_file=None, max_depth= None, feature_names=subset_attributes, class_names=None, label="all", filled=True, leaves_parallel=False, impurity=True, node_ids=False, proportion=True, rotate=False, rounded=True, special_characters=True, precision=3)
graph = graphviz.Source(dot_data)
graph.render("tree")