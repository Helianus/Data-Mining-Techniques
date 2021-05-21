from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

k = 15 # a constant number, you can change it

def gaussian_naive_bayes(Xtrain, Xtest, ytrain, ytest, flag):
    """
    Gaussian Naive Bayes method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    gnb = GaussianNB()
    model = gnb.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("Gaussian Naive Bayes: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred

def decision_tree(Xtrain, Xtest, ytrain, ytest, flag):
    """
    Decision Tree method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    clf = DecisionTreeClassifier(random_state=0)
    model = clf.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("Decision Tree: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred


def logistic_regression(Xtrain, Xtest, ytrain, ytest, flag):
    """
    Logistic Regression method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    lr = LogisticRegression(random_state=0)
    model = lr.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("Logistic Regression: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred

def k_nearest_neighbours(Xtrain, Xtest, ytrain, ytest, flag):
    """
    K Nearest Neighbours method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    knn = KNeighborsClassifier()
    model = knn.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("K Nearest Neighbours: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred


def support_vector_machine(Xtrain, Xtest, ytrain, ytest, flag):
    """
    Support Vector Machine method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    svm = SVC(kernel="linear", C=0.025, random_state=0)
    model = svm.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("Support Vector Machine: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred

def stochastic_gradient_descent(Xtrain, Xtest, ytrain, ytest, flag):
    """
    Stochastic Gradient Descent method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    sgd = SGDClassifier("modified_huber", shuffle=True, random_state=101)
    model = sgd.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("Stochastic Gradient Descent: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred


def xg_boost(Xtrain, Xtest, ytrain, ytest, flag):
    """
    XGBoost method
    
    param：
        Xtrain: input train data
        Xtest: input test data
        ytrain: output train data
        ytest: output test data
        flag: True for print accuracy score, False for not

    return:
        ypred: result of the predict model
    """
    xgb = XGBClassifier()
    model = xgb.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    accscore = accuracy_score(ytest, ypred).mean()
    evalmark = cross_val_score(model, Xtrain, ytrain, cv=k, scoring="accuracy").mean()
    if flag == "True":
        print("XGBoost: cross vali score", evalmark, ", accuracy score", accscore)
    
    return ypred