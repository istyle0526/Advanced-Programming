import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

xdata = pd.read_csv('./Data/EAGLE_P5_X.csv', header = None)
ydata = pd.read_csv('./Data/EAGLE_P5_Y.csv', header = None)
data = xdata.merge(ydata,left_index=True, right_index=True)


labels = ["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13",'y']
fig, ax = plt.subplots(figsize=(15,5))
corr = data.corr()

hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, xticklabels=labels, yticklabels=labels, cmap="coolwarm", fmt='.2f', linewidths=.05)
fig.subplots_adjust(left=.15, bottom=0.25)
plt.tight_layout()
plt.show()