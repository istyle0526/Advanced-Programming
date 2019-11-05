import xgboost as xgs
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


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
new_df.columns = range(new_df.shape[1])

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

MSE = []
for i in range(1,10):
    xgb = xgs.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=i)

    xgb.fit(train_x,train_y)
    predictions = xgb.predict(test_x)

    train_perf = r2_score(train_y, xgb.predict(train_x))
    test_perf = r2_score(test_y, xgb.predict(test_x))

    train_MSE = mean_squared_error(train_y, xgb.predict(train_x))
    test_MSE = mean_squared_error(test_y, xgb.predict(test_x))
    MSE.append(test_MSE)

max_depth = np.argmin(MSE)+1
xgb = xgs.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                       colsample_bytree=1, max_depth=max_depth)

xgb.fit(train_x, train_y)
predictions = xgb.predict(test_x)
print(explained_variance_score(test_y, predictions))

train_perf = r2_score(train_y, xgb.predict(train_x))
test_perf = r2_score(test_y, xgb.predict(test_x))

train_MSE = mean_squared_error(train_y, xgb.predict(train_x))
test_MSE = mean_squared_error(test_y, xgb.predict(test_x))

print("MSE of Train data is %.4f" %(train_MSE))
print("MSE of test data is %.4f" %(test_MSE))
print("R2 of Train data is %.4f" %(train_perf))
print("R2 of Test data is %.4f" %(test_perf))



#########################
