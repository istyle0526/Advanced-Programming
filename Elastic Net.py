from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
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
regr = ElasticNet(random_state=0)
regr.fit(train_x, train_y)

print(regr.coef_)

train_perf = r2_score(train_y, regr.predict(train_x))
test_perf = r2_score(test_y, regr.predict(test_x))

train_MSE = mean_squared_error(train_y, regr.predict(train_x))
test_MSE = mean_squared_error(test_y, regr.predict(test_x))


print("MSE of Train data is %.4f" %(train_MSE))
print("MSE of test data is %.4f" %(test_MSE))
print("R2 of Train data is %.4f" %(train_perf))
print("R2 of Test data is %.4f" %(test_perf))

