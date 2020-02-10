import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  ,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



dataset1 = pd.read_csv('H:/Software/Machine learning/Dataset/Bangladesh Weather/daily_rainfall_data(work).csv')
dataset1 = dataset1.drop(['StationIndex'] , axis = 1)


print(dataset1.memory_usage())
print('Dataset Size is {0} MB'.format(dataset1.memory_usage().sum()/1024**2))


labelencoder_X = LabelEncoder()
dataset1['Station']= labelencoder_X.fit_transform(dataset1['Station'])

gc.collect()


x = dataset1.drop(['Rainfall'] , axis = 1)
y = dataset1['Rainfall'].copy()



from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
new_df = sc_x.fit_transform(x)

x = pd.DataFrame(data=new_df, index=list(range(len(new_df))), columns=x.columns)

x_train , x_test , y_train , y_test = train_test_split(x.values , y.values , test_size = 0.25 , random_state = 0)


knn = KNeighborsRegressor(n_neighbors = 100)
knn.fit(x_train , y_train)

y_pred_knn = knn.predict(x_test)

ac = knn.score(x_test , y_test)
print(ac*100)



from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(x_train,y_train)

ridge_reg.score(x_test, y_test)


dt = DecisionTreeRegressor(max_depth = 12 , min_samples_leaf = 18)
dt.fit(x_train , y_train)


y_pred_dt = dt.predict(x_test)

ac = dt.score(x_test , y_test)
print(ac*100)




rf = RandomForestRegressor(n_estimators = 100 , random_state = 0)
rf.fit(x_train , y_train)

y_pred_rf = rf.predict(x_test)

ac_rf = r2_score(y_test , y_pred_rf)
print(ac_rf*100)


n_folds = 5
parameters = {'max_features':[1,2,3,4]}

rf = RandomForestRegressor(n_estimators = 100 , max_depth = 8)
grd_search = GridSearchCV(rf , parameters , cv = n_folds)

grd_search.fit(x_train, y_train)

score = grd_search.cv_results_
pd.DataFrame(score).head()



plt.figure()
plt.plot(score["param_max_features"],score["mean_train_score"],label="Training accuracy")
plt.plot(score["param_max_features"],score["mean_test_score"],label="Test accuracy")
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.show()
