import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

dataset = pd.read_csv('H:/Software/Machine learning/1/11.Netflix Real world Case Study for Multiple Linear Regression/dataset/mediacompany.csv')

dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date


from datetime import date

d0 = date(2017, 2, 28)
d1 = dataset.Date
delta = d1 - d0
dataset['day']= delta


dataset['day'] = dataset['day'].astype(str)
dataset['day'] = dataset['day'].map(lambda x: x[0:2])
dataset['day'] = dataset['day'].astype(int)


dataset.plot.line(x='day', y='Views_show')


fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel("Day")
host.set_ylabel("View_Show")
par1.set_ylabel("Ad_impression")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(dataset.day,dataset.Views_show, color=color1,label="View_Show")
p2, = par1.plot(dataset.day,dataset.Ad_impression,color=color2, label="Ad_impression")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))      
# no x-ticks                 
par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')


dataset['weekday'] = (dataset['day']+3)%7
dataset.weekday.replace(0,7, inplace=True)
dataset['weekday'] = dataset['weekday'].astype(int)



x = dataset[['Visitors' , 'weekday']]
y = dataset['Views_show'] 

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x , y)


x = sm.add_constant(x)
lm_1 = sm.OLS(y , x).fit()
print(lm_1.summary())



def cond(i):
    if i % 7 == 5:
        return 1
    elif i % 7 ==4:
        return 1
    else:
        return 0
    return i


dataset['weekend'] = [cond(i) for i in dataset['day']]
        

X = dataset[['Visitors','weekend','Character_A' , 'Ad_impression']]
y = dataset['Views_show']


X = sm.add_constant(X)

lm_2 = sm.OLS(y,X).fit()
print(lm_2.summary())






