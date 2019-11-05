from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy import stats


xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)

#ss = StandardScaler()
#xdata = ss.fit_transform(xdata)
#xdata = pd.DataFrame(xdata)

ndata = xdata.shape[0]
train_index=range(700)
test_index=range(700, ndata)

data = xdata.merge(ydata,left_index=True, right_index=True)
train_data = data.iloc[train_index,]

new_df =  train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]

train_x, train_y = new_df.iloc[:,:-1], new_df.iloc[:,-1]
test_x, test_y = xdata.iloc[test_index,], ydata.iloc[test_index,]


regr = linear_model.Lasso(alpha=4, normalize= False)
regr.fit(train_x,train_y)

train_perf = r2_score(train_y, regr.predict(train_x))
test_perf = r2_score(test_y, regr.predict(test_x))

train_MSE = mean_squared_error(train_y, regr.predict(train_x))
test_MSE = mean_squared_error(test_y, regr.predict(test_x))


print("MSE of Train data is %.4f" %(train_MSE))
print("MSE of test data is %.4f" %(test_MSE))
print("R2 of Train data is %.4f" %(train_perf))
print("R2 of Test data is %.4f" %(test_perf))
