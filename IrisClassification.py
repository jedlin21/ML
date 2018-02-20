from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Set KNN
knn = KNeighborsClassifier(n_neighbors  = 5)
knn.fit(X, y)

# Set Logistic regression
logreg = LogisticRegression()
logreg.fit(X, y)

#Make predict for KNN
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

#Make predict for Logistic regression
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))

#Split X and y into training and test examples
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#Train knn and logreg on splitted set
y_pred = knn.fit(x_train, y_train)
logreg.fit(x_train, y_train)

#Make predictions
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))


a = [[3,5,4,2],[3,5,4,1]]
print(knn.predict(a))
print(logreg.predict(a))
