from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

MM= MinMaxScaler(feature_range=(1, 100))

xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)

ndata = xdata.shape[0]
train_index=range(700)
test_index=range(700, ndata)

data = xdata.merge(ydata,left_index=True, right_index=True)
train_data = data.iloc[train_index,]
train_data = pd.DataFrame(MM.fit_transform(train_data.iloc[:,:-1]))

train_data = pd.DataFrame(np.log(train_data.iloc[:,:]))

new_df =  train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]

new_df = new_df.merge(ydata,left_index=True, right_index=True)
train_x, train_y = new_df.iloc[:,:-1], new_df.iloc[:,-1]
test_x, test_y = xdata.iloc[test_index,], ydata.iloc[test_index,]

alpha_subsets = [0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10]
mse=[]
mse_train = []

for i in alpha_subsets:
    clf = linear_model.Lasso(alpha=i, normalize= False)
    clf.fit(train_x,train_y)

    train_perf = r2_score(train_y, clf.predict(train_x))
    test_perf = r2_score(test_y, clf.predict(test_x))

    train_MSE = mean_squared_error(train_y, clf.predict(train_x))
    test_MSE = mean_squared_error(test_y, clf.predict(test_x))
    mse_train.append(train_MSE)
    mse.append(test_MSE)

fig, ax = plt.subplots()
ax.plot(alpha_subsets ,mse_train, color ='black', marker='o', label = 'MSE of train')
ax.plot(alpha_subsets,mse, color ='blue', marker='x', label = 'MSE of test')
plt.legend()
plt.xlabel('Alpha')
plt.ylabel('MSE')
ax.set(xlim=[0,11], ylim=[np.min([mse,mse_train])-5000, np.max([mse,mse_train])+5000])

optimal_alpha = alpha_subsets[np.argmin(mse)]

regr = linear_model.Lasso(alpha=optimal_alpha, normalize= False)
regr.fit(train_x,train_y)

train_perf = r2_score(train_y, regr.predict(train_x))
test_perf = r2_score(test_y, regr.predict(test_x))

train_MSE = mean_squared_error(train_y, regr.predict(train_x))
test_MSE = mean_squared_error(test_y, regr.predict(test_x))


print("MSE of Train data is %.4f" %(train_MSE))
print("MSE of test data is %.4f" %(test_MSE))
print("R2 of Train data is %.4f" %(train_perf))
print("R2 of Test data is %.4f" %(test_perf))

new_xdata = pd.read_csv('./Data/EAGLE_P5_X_new.csv', header = None)
estimates = pd.DataFrame(regr.predict(new_xdata))
estimates.to_csv('./Data/EAGLE_P5_Y_new.csv_2jo', sep=',',index=False, header=False)
