import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

def RMSLE(y,ypred):
    y=np.nan_to_num(y)
    ypred=np.nan_to_num(ypred)
    calc=(ypred-y)**2
    return np.sqrt(np.mean(calc))

res = pd.read_csv('train_data.csv')
y = res['rent_freq']
x = res.iloc[:,1:]
#x['datetime'] = x['datetime'].astype('category').cat.codes
x['icon'] = x['icon'].astype('category').cat.codes
x['holiday'] = x['holiday'].astype('category').cat.codes
x['weekday'] = x['weekday'].astype('category').cat.codes


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)


y_test_Log = np.log1p(y_test)
y_train_Log = np.log1p(y_train)


## Random Forest Regression


rmsle_scorer=sklearn.metrics.make_scorer(RMSLE,greater_is_better=False)

clf_4_cs=RandomForestRegressor()
param={'n_estimators':[200,300,400],'max_depth':[8,9,10]}
grid_4_cs=GridSearchCV(clf_4_cs,param_grid=param,scoring=rmsle_scorer,cv=5,n_jobs=4)
grid_4_cs.fit(x_train, y_train)
print ("Best params",grid_4_cs.best_params_)
print ("RMSLE score for casual train %f" %(RMSLE(y_train, grid_4_cs.predict(x_train))))
print ("RMSLE score for casual test %f" %(RMSLE(y_test, grid_4_cs.predict(x_test))))



OS = []
for i in range(100,500,50):
    rfr = RandomForestRegressor(n_estimators=300,max_depth=10)

    rfr.fit(x_train, y_train)
    score = rfr.score(x_test, y_test)
    OS.append(MSE)

plt.gca()
plt.plot(np.linspace(100,500,8),OS)
plt.xlabel("log C")
plt.ylabel("OS accuracy")
plt.title("Accuracy vs. penalization constant (log C)")
plt.xlim(100,500)
  



print(score = rfr.score(x_test, y_test))




## Gradient Boosting on Regression Tree

#plot the accuracy
N = np.logspace(3000, 5000, 100)
OS = []
for c in N:
    clf = GradientBoostingRegressor(n_estimators=c, alpha = 0.01) 
    clf.fit(x_train, y_train)
    MSE=RMSLE(y_train, clf.best_estimator_.predict(x_train))
    OS.append(MSE)

plt.gca()
plt.plot(np.linspace(-10,10,300),OS)
plt.xlabel("n_estimator")
plt.ylabel("OS accuracy")
plt.title("Accuracy vs. penalization constant ")
plt.xlim(-10,10)
plt.show()



gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
gbm.fit(x_train,y_train_Log)
preds = gbm.predict(X= x_test)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(y_test_Log),np.exp(preds),False))





#imshow(confusion_matrix(np.exp(y_test_Log),np.exp(preds)))


## SVR
#fine tuning
rmsle_scorer=sklearn.metrics.make_scorer(RMSLE,greater_is_better=False)

rg_svr=SVR()
param={'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel' :['rbf','sigmoid','linear']}
grid_4_cs=GridSearchCV(rg_svr,param_grid=param,scoring=rmsle_scorer,cv=5,n_jobs=4)
grid_4_cs.fit(x_train, y_train)
print ("Best params",grid_4_cs.best_params_)
print ("RMSLE score for casual train %f" %(RMSLE(y_train, grid_4_cs.predict(x_train))))
print ("RMSLE score for casual test %f" %(RMSLE(y_test, grid_4_cs.predict(x_test))))


#plot the accuracy
C = np.logspace(-3, 2, 6)
OS = []
for c in C:
    clf = svm.SVR(kernel='sigmoid',C=c) 
    clf.fit(x_train, y_train)
    MSE= RMSLE(y_train, grid_4_cs.best_estimator_.predict(x_train))
    OS.append(MSE)

plt.gca()
plt.plot(np.linspace(-10,10,300),OS)
plt.xlabel("log C")
plt.ylabel("OS accuracy")
plt.title("Accuracy vs. penalization constant (log C)")
plt.xlim(-10,10)
plt.show()



#need to change the kernel from Gaussian Kernel to RBM
svr = SVR()
y_train_Log = np.log1p(y_train)
svr.fit(x_train,y_train_Log)
y_test_Log = np.log1p(y_test)
preds = svr.predict(X= x_test)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(y_test_Log),np.exp(preds),False))


#imshow(confusion_matrix(np.exp(y_test_Log),np.exp(preds)))


## Cluster

