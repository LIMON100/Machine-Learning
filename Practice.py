import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



dataset1 = pd.read_csv('G:/Software/Machine learning/Datasets/Breast_Cancer/77_cancer_proteomes_CPTAC_itraq.csv')
data_text = pd.read_csv('G:/Software/Machine learning/1/26. Project  Kaggle/training/training_text.csv' , sep = "\|\|"  , engine = 'python' , names = ["ID" , "TEXT"] , skiprows = 1)
dataset2 = pd.read_csv('G:/Software/Machine learning/Datasets/Breast_Cancer/breast-cancer.csv')


dataset1.isnull().sum()
dataset2.isnull().sum()


print('Dataset Size is {0} MB'.format(dataset1.memory_usage().sum()/1024**2))

dataset1 = dataset1.fillna(method = 'ffill')
print(np.unique(dataset1['gene_symbol']))


dataset1 = dataset1.fillna(np.mean(dataset1))

dataset1 = dataset1[~np.isnan(dataset1).any(axis=1)]

dataset1 = dataset1.dropna()

dataset2['diagnosis'] = dataset2['diagnosis'].map({'M':1 , 'B':0})


x_train , x_test , y_train , y_test = train_test_split()

sc = StandardScaler()
df = sc.fit_transform(dataset2)