import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('H:/Software/Machine learning/1.1/Machine Learning A-Z™ Hands-On Python & R In Data Science/12 Logistic Regression/Code/Logistic_Regression/Social_Network_Ads.csv')

plt.subplot(2,2,1)
sns.distplot(dataset['Age'])

plt.subplot(2,2,2)
sns.distplot(dataset['EstimatedSalary'] )

plt.subplot(2,2,3)
sns.distplot(dataset['Purchased'])

sns.barplot(x = 'EstimatedSalary',  y = 'Purchased' , data = dataset)