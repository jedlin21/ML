from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors  = 1)
knn.fit(X, y)

a = [[3,5,4,2],[3,5,4,1]]

print(a)
print(knn.predict(a))
