import pandas as pd
import seaborn as sns

dataset = pd.read_csv('H:/Software\Machine learning/5/Machine Learning A-Z™ Hands-On Python & R In Data Science/04 Simple Linear Regression/Simple_Linear_Regression/Salary_Data.csv')


sns.heatmap(dataset.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='blue',annot=True)