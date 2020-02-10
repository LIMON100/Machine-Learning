import matplotlib.pyplot as plt
import pandas as pd

#dataset = pd.read_csv('salary.csv')

dataset = pd.read_csv('H:/Software/Machine learning/Dataset/Linear Regression/tvmarketing.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


print(regressor.score(X_test , y_test))


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'red')


plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('X vs Y (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



plt.scatter(y_test, y_pred, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('X vs Y (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()