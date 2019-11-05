from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)

ss = StandardScaler()
xdata = ss.fit_transform(xdata)
xdata = pd.DataFrame(xdata)


ndata = xdata.shape[0]
train_index=range(700)
test_index=range(700, ndata)

new_df =  train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]

train_x, train_y = new_df.iloc[:,:-1], new_df.iloc[:,-1]
test_x, test_y = xdata.iloc[test_index,], ydata.iloc[test_index,]


k=1
pca = PCA().fit(train_x)
for i in range(0,len(np.cumsum(pca.explained_variance_ratio_))):
    if np.cumsum(pca.explained_variance_ratio_)[i]>0.80:
        k=i+1
        break

print("optimal number of selected Principal Components is %.f"%(k))

pca = PCA(n_components=k)
PC_train = pca.fit_transform(train_x)
PC_test = pca.transform(test_x)


model = LinearRegression()
model.fit(PC_train, train_y)
train_perf = r2_score(train_y, model.predict(PC_train))
test_perf = r2_score(test_y, model.predict(PC_test))
print(model.coef_)

print("학습 데이터로 측정한 모델의 성능은 %.4f 입니다." %(train_perf))
print("테스트 데이터로 측정한 모델의 성능은 %.4f 입니다." %(test_perf))

log=LogisticRegression()
log.fit(PC_train,train_y)
estimates = log.predict(PC_test)
estimates_2 = log.predict(PC_train)

C=confusion_matrix(Y_test,estimates)
TN, FP, FN, TP = C.ravel()

Accuracy= log.score(PC_test,Y_test)
Precision=float(TP/(TP+FP))
Recall=float(TP/(TP+FN))
Specificity=float(TN/(TN+FP))
F1measure=float(2*Precision*Recall/(Precision+Recall))
Gmean=float(np.sqrt(Precision*Recall))

print("\n"
      "\n"
      "This solution is computed using test data")
print(C)
print("Accuracy using test data is: %.3f"%(Accuracy))
print("Precision : %.3f, Recall : %.3f, Specificity : %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision, Recall, Specificity, F1measure, Gmean))
print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity, 1-Recall))


C_2 =confusion_matrix(Y_train,estimates_2)
TN_2, FP_2, FN_2, TP_2 = C_2.ravel()

accuracy_2=float(log.score(PC_train,Y_train))
Precision_2=float(TP_2/(TP_2+FP_2))
Recall_2=float(TP_2/(TP_2+FN_2))
Specificity_2=float(TN_2/(TN_2+FP_2))
F1measure_2=float(2*Precision_2*Recall_2/(Precision_2+Recall_2))
Gmean_2=float(np.sqrt(Precision_2*Recall_2))

print("\n"
      "\n"
      "This solution is computed using train data")
print(C_2)
print("Accuracy using train data is: %.3f"%(accuracy_2))
print("Precision : %.3f, Recall : %.3f, Specificity: %.3f, F1measure : %.3f, G-mean : %.3f" %(Precision_2, Recall_2, Specificity_2, F1measure_2, Gmean_2))
print("Type 1 error : %.3f, Type 2 error : %.3f"%(1-Specificity_2, 1-Recall_2))

data =xdata.merge(ydata,left_index = True, right_index= True)

import pymrmr