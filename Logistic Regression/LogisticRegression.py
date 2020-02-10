import numpy as np
import pandas as pd

churnData = pd.read_csv('H:/Software/Machine learning/1/17. Logistic Regression/LogisticReg/churn_data.csv')
customerData = pd.read_csv('H:/Software/Machine learning/1/17. Logistic Regression/LogisticReg/customer_data.csv')
internetData = pd.read_csv('H:/Software/Machine learning/1/17. Logistic Regression/LogisticReg/internet_data.csv')


#Merger data or Merging the all information

df1 = pd.merge(churnData , customerData , how = 'inner' , on = 'customerID')
telecom = pd.merge(df1 , internetData , how = 'inner' , on = 'customerID')


#Data Prepoaration or Put Yes=1 and No=0

telecom['PhoneService'] =  telecom['PhoneService'].map({'Yes': 1 , 'No': 0})
telecom['PaperlessBilling'] =  telecom['PaperlessBilling'].map({'Yes': 1 , 'No': 0})
telecom['Churn'] =  telecom['Churn'].map({'Yes': 1 , 'No': 0})
telecom['Partner'] =  telecom['Partner'].map({'Yes': 1 , 'No': 0})
telecom['Dependents'] =  telecom['Dependents'].map({'Yes': 1 , 'No': 0})



#Dummy Variable

cont = pd.get_dummies(telecom['Contract'],prefix='Contract',drop_first=True)
telecom = pd.concat([telecom,cont],axis=1)


pm = pd.get_dummies(telecom['PaymentMethod'] , prefix = 'PaymentMethod' , drop_first = True)
telecom = pd.concat([telecom , pm] , axis = 1)


gen = pd.get_dummies(telecom['gender'],prefix='gender',drop_first=True)
telecom = pd.concat([telecom,gen],axis=1)


ml = pd.get_dummies(telecom['MultipleLines'],prefix='MultipleLines')
ml1 = ml.drop(['MultipleLines_No phone service'],1)
telecom = pd.concat([telecom,ml1],axis=1)



iser = pd.get_dummies(telecom['InternetService'],prefix='InternetService',drop_first=True)
telecom = pd.concat([telecom,iser],axis=1)



os = pd.get_dummies(telecom['OnlineSecurity'],prefix='OnlineSecurity')
os1= os.drop(['OnlineSecurity_No internet service'],1)
telecom = pd.concat([telecom,os1],axis=1)



ob =pd.get_dummies(telecom['OnlineBackup'],prefix='OnlineBackup')
ob1 =ob.drop(['OnlineBackup_No internet service'],1)
telecom = pd.concat([telecom,ob1],axis=1)


 
dp =pd.get_dummies(telecom['DeviceProtection'],prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'],1)
telecom = pd.concat([telecom,dp1],axis=1)



ts =pd.get_dummies(telecom['TechSupport'],prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'],1)
telecom = pd.concat([telecom,ts1],axis=1)



st =pd.get_dummies(telecom['StreamingTV'],prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'],1)
telecom = pd.concat([telecom,st1],axis=1)


 
sm =pd.get_dummies(telecom['StreamingMovies'],prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'],1)
telecom = pd.concat([telecom,sm1],axis=1)


#Dropping The repeated variable

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies'], 1)


#Cheking Outliees

num_telecom = telecom[['tenure' , 'MonthlyCharges' , 'SeniorCitizen' , 'TotalCharges']]
num_telecom.describe(percentiles=[.25,.5,.75,.90,.95,.99])



#Cheking Missing values


#Feature Standardisation

df = telecom[['tenure','MonthlyCharges','TotalCharges']]
normalized_df = (df-df.mean()) / df.std()
telecom = telecom.drop(['tenure','MonthlyCharges','TotalCharges'], 1)
telecom = pd.concat([telecom,normalized_df],axis=1)


#Check the Churn Rate

churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100


#Splitting Data

from sklearn.model_selection import train_test_split

X = telecom.drop(['Churn','customerID'],axis=1)
y = telecom['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.25,random_state=100)



#Running First Model

import statsmodels.api as sm

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()




#Correlation Matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (20,10))
sns.heatmap(telecom.corr(),annot = True)




#Dropping Highly Correlated variables

X_test2 = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)
X_train2 = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)




#Chekcing the Correlation Matrix

plt.figure(figsize = (20,10))
sns.heatmap(X_train2.corr(),annot = True)




#Re-running the MODEL


logm2 = sm.GLM(y_train,(sm.add_constant(X_train2)), family = sm.families.Binomial())
logm2.fit().summary()



#Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 13)             
rfe = rfe.fit(X,y)
print(rfe.support_)
print(rfe.ranking_)    


col = ['PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year','PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No','OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logsk = LogisticRegression()
logsk.fit(X_train[col], y_train)


logm4 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())
logm4.fit().summary()


def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


telecom.columns
['PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']




vif_cal(input_data=telecom.drop(['customerID','SeniorCitizen', 'Partner', 'Dependents',
                                 'PaymentMethod_Credit card (automatic)','PaymentMethod_Mailed check',
                                 'gender_Male','MultipleLines_Yes','OnlineSecurity_No','OnlineBackup_No',
                                 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',
                                 'TechSupport_No','StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',
                                 'MonthlyCharges'], axis=1), dependent_col='Churn')








#Dropping variable with high VIF


col = ['PaperlessBilling', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']


logm5 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())
logm5.fit().summary()


vif_cal(input_data=telecom.drop(['customerID','PhoneService','SeniorCitizen', 'Partner', 'Dependents',
                                 'PaymentMethod_Credit card (automatic)','PaymentMethod_Mailed check',
                                 'gender_Male','MultipleLines_Yes','OnlineSecurity_No','OnlineBackup_No',
                                 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',
                                 'TechSupport_No','StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',
                                 'MonthlyCharges'], axis=1), dependent_col='Churn')



from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logsk = LogisticRegression()
logsk.fit(X_train[col], y_train)



#MAKING PREDICTION

y_pred = logsk.predict_proba(X_test[col])
y_pred_df = pd.DataFrame(y_pred)
y_pred_1 = y_pred_df.iloc[:,[1]]


y_test_df = pd.DataFrame(y_test)

y_test_df['CustID'] = y_test_df.index

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)

y_pred_final= y_pred_final.rename(columns={ 1 : 'Churn_Prob'})

y_pred_final = y_pred_final.reindex_axis(['CustID','Churn','Churn_Prob'], axis=1)





#Model EVALUTION

confusion = metrics.confusion_matrix( y_pred_final.Churn, y_pred_final.predicted )

metrics.accuracy_score( y_pred_final.Churn, y_pred_final.predicted)


TP = confusion[1,1] 
TN = confusion[0,0] 
FP = confusion[0,1] 
FN = confusion[1,0]


TP / float(TP+FN)
TN / float(TN+FP)

print(FP/ float(TN+FP))
print (TP / float(TP+FP))
print (TN / float(TN+ FN))



#ROC Curve

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds

draw_roc(y_pred_final.Churn, y_pred_final.predicted)


