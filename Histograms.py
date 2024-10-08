import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

col_names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv',names=col_names)

plt.clf()
#히스토그램
# data.hist(figsize=(12,10), bins=5)
# plt.tight_layout()
# plt.savefig('./results/histograms.png')

#박스플롯
data.plot(kind='box', subplots=True, figsize=(12,10), layout=(3,3), sharex=False, sharey=False)
plt.savefig('./results/boxplot.png')

#선그래프
# data.plot(kind='density', figsize=(12,10), subplots=True, layout=(3,3), sharex=False)
# plt.savefig('./results/density_plot.png')

#산점도
# pd.plotting.scatter_matrix(data)
# plt.savefig('./results/scatter.png')