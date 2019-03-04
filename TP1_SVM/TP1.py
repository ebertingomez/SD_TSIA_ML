from sklearn.svm import SVC, LinearSVC
import numpy as np
import matplotlib.pyplot as plt

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

print("=======================")


print("SVM Linear Train Errors: ", np.sum(np.abs(y_train_lin - y_train)))
print("SVM Linear Test Errors: ", np.sum(np.abs(y_pred_lin - y_test)))

print("=======================")

clf_linSVC = LinearSVC()
clf_linSVC.fit(X_train, y_train)

y_train_linSVC = clf_linSVC.predict(X_train)
y_pred_linSVC = clf_linSVC.predict(X_test)

print("Linear SVC Train Errors: ", np.sum(np.abs(y_train_linSVC - y_train)))
print("Linear SVC Test Errors: ", np.sum(np.abs(y_pred_linSVC - y_test)))

for i in np.arange(1,6,1):
    print("=======================")
    print("Poly SVC with degree: ", i)
    clf_pol = SVC(degree=i,kernel='poly',coef0=0, gamma='scale')
    clf_pol.fit(X_train, y_train)

    y_train_pol = clf_pol.predict(X_train)
    y_pred_pol = clf_pol.predict(X_test)
    print("\tTrain Errors: ", np.sum(np.abs(y_train_pol - y_train)))
    print("\tTest Errors: ", np.sum(np.abs(y_pred_pol - y_test)))


###################################

fignum = 1

clf_lin = SVC(kernel='linear')
clf_lin.fit(X_train, y_train)
plt.figure(fignum, figsize=(4, 3))
plt.clf()

plt.scatter(clf_lin.support_vectors_[:, 0], clf_lin.support_vectors_[:, 1], s=80,
            facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

plt.axis('tight')
x_min = np.amin(X_train,0)[0]
x_max = np.amax(X_train,0)[0]
y_min = np.amin(X_train,0)[1]
y_max = np.amax(X_train,0)[1]


XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf_lin.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(4, 3))
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
            levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
fignum = fignum + 1

for alpha in np.arange(1,6,1):
    clf_pol = SVC(degree=alpha,kernel='poly',coef0=1,gamma='scale')
    clf_pol.fit(X_train, y_train)
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf_pol.support_vectors_[:, 0], clf_pol.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf_pol.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()

def printSVM(clf,X,y,fignum):
    clf_pol.fit(X, y)
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf_pol.support_vectors_[:, 0], clf_pol.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')

    x_min = np.amin(X,0)[0]
    x_max = np.amax(X,0)[0]
    y_min = np.amin(X,0)[1]
    y_max = np.amax(X,0)[1]

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf_pol.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())