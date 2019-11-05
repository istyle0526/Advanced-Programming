import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

rawData = np.genfromtxt("./Data/EAGLE_P5_X.csv", delimiter=",")
ss = StandardScaler()
scaledData = ss.fit_transform(rawData)
yData= np.genfromtxt("./Data/EAGLE_P5_Y.csv", delimiter=",")
ndata = rawData.shape[0]


xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)

ss = StandardScaler()
xdata = ss.fit_transform(xdata)
xdata = pd.DataFrame(xdata)


ndata = xdata.shape[0]
train_index=range(700)
test_index=range(700, ndata)

data = xdata.merge(ydata,left_index=True, right_index=True)
train_data = data.iloc[train_index,]
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
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

train_mse = []
test_mse = []


for k in range (1,20,2):
    KNN_R = KNeighborsRegressor(n_neighbors=k, p=2)
    KNN_R.fit(train_x, train_y)
    estimates = KNN_R.predict(train_x)
    estimates2 = KNN_R.predict(test_x)

    mse_train = mean_squared_error(estimates, train_y)
    mse_test = mean_squared_error(estimates2, test_y)

    r2_train = r2_score(estimates, train_y)
    r2_test = r2_score(estimates2, test_y)

    train_mse.append(mse_train)
    test_mse.append(mse_test)


optimal_k = 2*np.argmin(test_mse)+1

KNN_R = KNeighborsRegressor(n_neighbors=optimal_k, p=2)
KNN_R.fit(train_x, train_y)
estimates = KNN_R.predict(train_x)
estimates2 = KNN_R.predict(test_x)

mse_train = mean_squared_error(train_y,estimates)
mse_test = mean_squared_error(test_y, estimates2)

r2_train = r2_score(train_y, estimates)
r2_test = r2_score(test_y,estimates2)

print("*****************************************")
print("Optimal K is %d"%optimal_k)
print("Train : MSE on %d neighbors : %.4f" % (optimal_k, mse_train))
print("Train : r2score on %d neighbors : %.4f" % (optimal_k, r2_train))
print("Test : MSE on %d neighbors : %.4f" % (optimal_k, mse_test))
print("Test : r2 score on %d neighbors : %.4f" % (optimal_k, r2_test))