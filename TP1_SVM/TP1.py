from sklearn.svm import SVC
import numpy as np

from sklearn.model_selection import train_test_split

# Load iris dataset

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.20, random_state=42)

clf_lin = SVC(kernel='linear')
clf_lin.fit(X_train, y_train)

y_train_lin = clf_lin.predict(X_train)
y_pred_lin = clf_lin.predict(X_test)

print("Train Errors: ", np.sum(np.abs(y_train_lin - y_train)))
print("Test Errors: ", np.sum(np.abs(y_pred_lin - y_test)))


###################################

clf_pol = SVC(gamma=100,kernel='poly',coef0=5)
clf_pol.fit(X_train, y_train)

y_train_pol = clf_pol.predict(X_train)
y_pred_pol = clf_pol.predict(X_test)

print("Train Errors: ", np.sum(np.abs(y_train_pol - y_train)))
print("Test Errors: ", np.sum(np.abs(y_pred_pol - y_test)))
