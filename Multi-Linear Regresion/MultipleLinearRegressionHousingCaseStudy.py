import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Multiple Linear Rregression/Housing.csv')

#sns.pairplot(dataset)


dataset['mainroad'] = dataset['mainroad'].map({'yes': 1, 'no': 0})
dataset['guestroom'] = dataset['guestroom'].map({'yes': 1, 'no': 0})
dataset['basement'] = dataset['basement'].map({'yes': 1, 'no': 0})
dataset['hotwaterheating'] = dataset['hotwaterheating'].map({'yes': 1, 'no': 0})
dataset['airconditioning'] = dataset['airconditioning'].map({'yes': 1, 'no': 0})
dataset['prefarea'] = dataset['prefarea'].map({'yes': 1, 'no': 0})


status = pd.get_dummies(dataset['furnishingstatus'],drop_first=True)
dataset = pd.concat([dataset,status],axis=1)
dataset.drop(['furnishingstatus'] , axis=1 , inplace=True)

#dataset = (dataset - dataset.mean())/dataset.std()
dataset = (dataset-np.min(dataset)) / (np.max(dataset) - np.min(dataset))


x = dataset.drop(['price'] , axis = 1)
y = dataset['price']



#x['intercept'] = 1
#x = x.reindex_axis(['intercept','area','bedrooms' , 'bathrooms' , 'stories' , 'mainroad' , 'basement' , 'parking'], axis=1)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)


x_train = sm.add_constant(x_train) 
lm_1 = sm.OLS(y_train , x_train).fit()
lm_1.params
lm_1.summary()


def vif_call(input_data , dependent_col):
    vif_val = pd.DataFrame(columns = ['Features' , 'Vif'])
    x_var = input_data.drop([dependent_col] , axis = 1)
    x_var_name = x_var.columns
    
    for i in range(0,x_var_name.shape[0]):
        y=x_var[x_var_name[i]] 
        x=x_var[x_var_name.drop(x_var_name[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif = round(1/(1-rsq),2)
        vif_val.loc[i] = [x_var_name[i], vif]
        
    return vif_val.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)



print(vif_call(input_data=dataset , dependent_col="price"))


x_train = x_train.drop('bedrooms', 1)

lm_2 = sm.OLS(y_train,x_train).fit()
print(lm_2.summary())



x_test_new = sm.add_constant(x_test)
x_test_new = x_test_new.drop('bedrooms',1)



y_pred_new = lm_2.predict(x_test_new)


c = [i for i in range(1,165,1)]
fig = plt.figure()
plt.plot(c , y_test, color="blue", linewidth=2.5, linestyle="-")     
plt.plot(c , y_pred_new, color="red",  linewidth=2.5, linestyle="-")  
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Housing Price', fontsize=16)



fig = plt.figure()
plt.scatter(y_test,y_pred_new)
fig.suptitle('y_test vs y_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16)  



print("The R squared Error: ",r2_score(y_test , y_pred_new))