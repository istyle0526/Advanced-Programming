# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score

# contrived dataset
xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)

ndata = xdata.shape[0]
train_index=range(700)
test_index=range(700, ndata)

train_x, test_x = xdata.iloc[train_index,], xdata.iloc[test_index,]
train_y, test_y = ydata.iloc[train_index,], ydata.iloc[test_index,]

# fit model
model = ExponentialSmoothing(train_x.iloc[:,0], seasonal = 'mul', seasonal_periods=365).fit()
# make prediction

yhat = model.predict(start = 700, end = 899)
print(r2_score(test_x.iloc[:,0],model.predict()))