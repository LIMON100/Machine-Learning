from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

boston = load_boston()
print(boston.data.shape)

print(boston.feature_names)

boston = pd.DataFrame(boston.data)

sns.boxplot(x = boston[7])


boston_plot = boston

fig,ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_plot[2] , boston_plot[9])
plt.show()



boston_iqr = boston
Q1 = boston_iqr.quantile(0.25)
Q3 = boston_iqr.quantile(0.75)
iqr = Q3 - Q1
print(iqr)


print((boston_iqr < (Q1 - 1.5*iqr)) | (boston_iqr > (Q3 + 1.5*iqr)))


boston_iqr_clean = boston_iqr[~((boston_iqr < (Q1 - 1.5*iqr)) | (boston_iqr > (Q3 + 1.5*iqr))).any(axis = 1)]