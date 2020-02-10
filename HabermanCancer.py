import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Haberman Cancer/haberman.csv')

shape = dataset.shape
print(shape)

info = dataset.info()
print(info)

col = dataset.columns
print(col)


count = dataset['status'].value_counts()
print(count)


"""2-D scatter-plot"""


dataset.plot(kind='scatter', x='age', y='year') ;
plt.show()


dataset.plot(kind='scatter', x='age', y='nodes') ;
plt.show()


dataset.plot(kind='scatter', x='year', y='nodes') ;
plt.show()



sns.set_style('whitegrid')
sns.FacetGrid(dataset , hue = 'status' , size = 4)\
    .map(plt.scatter , "age" , "year")\
    .add_legend()
plt.show()


sns.set_style('whitegrid')
sns.FacetGrid(dataset , hue = 'status' , size = 4)\
    .map(plt.scatter , "age" , "nodes")\
    .add_legend()
plt.show()


sns.set_style('whitegrid')
sns.FacetGrid(dataset , hue = 'status' , size = 4)\
    .map(plt.scatter , "year" , "nodes")\
    .add_legend()
plt.show()


"""Objective
    
1) Find a good solution of separating the CLASSIFICATION"""

"""Observation
1) On the first examination we see that separating the OUTPUT based on age + year is hard
2) On the second examination we see that separating the OUTPUT based on age + nodes is very hard.They are ovelaping each other
3) On the third examination we see that separating the OUTPUT based on nodes + year is hard for overlaping
4) From all above experiment it is clear that,it is very hard to separate them.

    """
    

"""3-D scatter-plot"""
    
import plotly.express as px
fig = px.scatter_3d(dataset, x='age', y='year', z='nodes',color='status')
fig.show()






"""Pair-plot"""

sns.set_style("whitegrid")
sns.pairplot(dataset , hue = 'status' , size = 3)
plt.show()





"""Histogram, PDF, CDF"""


count_1 = dataset.iloc[dataset['status'] == '1']
count_2 = dataset.iloc[dataset['status'] == '2']

plt.plot(count_1['age'] , np.zeros_like(dataset['age']) , 'o')
plt.plot(count_2['age'] , np.zeros_like(dataset['age']) , 'o')

plt.show()





sns.FacetGrid(dataset, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();


sns.FacetGrid(dataset, hue="status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();


sns.FacetGrid(dataset, hue="status", size=5) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.show();





counts , bin_edges = np.histogram(dataset['age'] , bins = 10 , density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:] , pdf)
plt.plot(bin_edges[1:] , cdf)


counts , bin_edges = np.histogram(dataset['age'] , bins = 10 , density = True)

pdf = counts/sum(counts)
plt.plot(bin_edges[1:] , pdf)

plt.show()






counts, bin_edges = np.histogram(dataset['age'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# 1
counts, bin_edges = np.histogram(dataset['age'], bins=10,  density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#2
counts, bin_edges = np.histogram(dataset['age'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();








counts, bin_edges = np.histogram(dataset['year'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# 1
counts, bin_edges = np.histogram(dataset['year'], bins=10,  density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#2
counts, bin_edges = np.histogram(dataset['year'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();








counts, bin_edges = np.histogram(dataset['nodes'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# 1
counts, bin_edges = np.histogram(dataset['nodes'], bins=10,  density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#2
counts, bin_edges = np.histogram(dataset['nodes'], bins=10, density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();




"""Box-Plot"""
sns.boxplot(x='status',y='age', data=dataset)
plt.show()

sns.boxplot(x='status',y='year', data=dataset)
plt.show()

sns.boxplot(x='status',y='nodes', data=dataset)
plt.show()






"""violin-plot"""
sns.violinplot(x="status", y="age", data=dataset, size=4)
plt.show()



sns.violinplot(x="status", y="year", data=dataset, size=4)
plt.show()



sns.violinplot(x="status", y="nodes", data=dataset, size=4)
plt.show()






