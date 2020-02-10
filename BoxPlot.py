import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Iris.csv')

sns.boxplot(x='Species',y='PetalLengthCm', data=dataset)
plt.show()


sns.violinplot(x="Species", y="PetalLengthCm", data=dataset, size=8)
plt.show()