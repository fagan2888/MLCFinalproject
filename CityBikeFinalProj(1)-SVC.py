#!/usr/bin/env python
# coding: utf-8

# # Run codes with the notation "#Run"

# In[5]:


#Run
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorboardX import SummaryWriter
writer = SummaryWriter()


# In[80]:


#Run
#citiDataRaw=pd.read_csv("citibike_final_sample.csv")
citiDataRaw=pd.read_csv("citibike_final_sample.csv")
citiDataRaw.head()


# ##  Subtract the outlier for latitude

# In[85]:


#Run
citiDataRaw = citiDataRaw[citiDataRaw['endstationlatitude']<45 ]
citiDataRaw = citiDataRaw[citiDataRaw['startstationlatitude']<45 ]
citiDataRaw = citiDataRaw[citiDataRaw['startstationlongitude']< -73 ]
citiDataRaw = citiDataRaw[citiDataRaw['endstationlongitude']< -73]
                          # need to run
citiDataRaw = citiDataRaw[citiDataRaw['endstationlatitude']>40 ]
citiDataRaw = citiDataRaw[citiDataRaw['startstationlatitude']>40 ]
citiDataRaw = citiDataRaw[citiDataRaw['startstationlongitude']>-75 ]
citiDataRaw = citiDataRaw[citiDataRaw['endstationlongitude']>-75]


# In[86]:


citiDataRaw.endstationlatitude.sort_values(ascending=True).head(10)


# ## see the station dock

# **516 station with dock number, 825 total station. In the following code, we seperate the dock and then in phrase two, we put the dock back.**

# In[92]:


#For checking, do not need to run in whole dataframe
a = citiDataRaw[['startstationid','starttotaldocks']][citiDataRaw.starttotaldocks>0.0].groupby('startstationid').mean()
print(a.shape)
a.head()


# In[93]:


#check precipAccumulation
#For checking, do not need to run in whole dataframe
a = citiDataRaw[['startstationid','precipAccumulation']].groupby('startstationid').mean()
print(a.shape)
a.tail()


# In[95]:


#Run
#replace prepAccumulation NaN to 0
citiDataRaw['precipAccumulation'] = citiDataRaw.precipAccumulation.replace(np.nan, 0)
a = citiDataRaw[['startstationid','precipAccumulation']].groupby('startstationid').mean()
a.tail()


# In[101]:


#Run
#replace prepAccumulation NaN to 0
#citiDataRaw[''] = citiDataRaw.precipAccumulation.replace(np.nan, 0)
a = citiDataRaw[['startstationid','precipIntensity']].groupby('startstationid').mean()
a.tail()


# ## get zipcode

# In[104]:


zipcode = pd.read_csv('citibikezipcode.csv')
zipcode.head()


# ## Get the data we need

# In[58]:


citiDataRaw.columns


# In[ ]:





# In[107]:


cols = []
CitiBikeData = citiDataRaw[['starttime','stoptime', 'startstationid', 'endstationid','startdate', 'datetime',
                           'apparentTemperature', 'cloudCover', 'humidity', 'icon' ,'precipAccumulation',
                            'precipIntensity', 'temperature', 'uvIndex', 'visibility', 'windSpeed',
                           'date', 'Holiday', 'year', 'month', 'weekday', 'starthour', 'endhour']]



CitiBikeData.columns = ['start_time','stop_time', 'start_station_id', 'end_station_id','start_date', 'datetime',
                           'apparent_temperature', 'cloud_cover', 'humidity', 'icon' ,'precipAccumulation',
                            'precip_intensity', 'temperature', 'uv_index', 'visibility', 'wind_speed',
                           'date', 'holiday', 'year', 'month', 'weekday', 'start_hour', 'end_hour']
CitiBikeData.dropna(inplace = True)

# print(CitiBikeData.dtypes)
CitiBikeData.head()

#CitiBikeData.describe()
# CitiBikeData.isnull().sum(axis = 0)


# In[108]:


CitiBikeData['start_time']= pd.to_datetime(CitiBikeData['start_time'])
CitiBikeData['stop_time']= pd.to_datetime(CitiBikeData['stop_time'])
# CitiBikeData['datetime']= pd.to_datetime(CitiBikeData['datetime'])

CitiBikeData['start_station_id'] = CitiBikeData['start_station_id'].astype('int64')
CitiBikeData['end_station_id'] = CitiBikeData['end_station_id'].astype('int64')
print(CitiBikeData.shape)
CitiBikeData.head()


# In[137]:


new = pd.merge(CitiBikeData,zipcode,right_on = 'startstatid',left_on = 'start_station_id')
print(new.columns)
new.head()


# In[138]:


new = new[new.NJ==0]
new.shape


# In[139]:


rentFreq = CitiBikeData[['start_station_id', 'datetime']]
rentFreq = rentFreq.groupby(['start_station_id', 'datetime']).size().to_frame().reset_index()

returnFreq = CitiBikeData[['end_station_id', 'datetime']]
returnFreq = returnFreq.groupby(['end_station_id', 'datetime']).size().reset_index()


# In[140]:


returnFreq.columns = ['station_id', 'datetime', 'return_freq']
rentFreq.columns = ['station_id', 'datetime', 'rent_freq']


# In[141]:


res = pd.merge(rentFreq, 
               CitiBikeData, 
               left_on=['station_id', 'datetime'], 
               right_on=['start_station_id', 'datetime'])

res = pd.merge(returnFreq, 
               res, 
               left_on=['station_id', 'datetime'], 
               right_on=['end_station_id', 'datetime'])


# In[143]:


cols = [ 'rent_freq', 'start_station_id', 'return_freq', 
                           'apparent_temperature', 'cloud_cover', 'humidity', 'icon' ,'precipAccumulation',
                            'precip_intensity', 'temperature', 'uv_index', 'visibility', 'wind_speed',
                            'holiday',  'weekday']

# res.columns
res = res[cols]
res.drop_duplicates()


# In[144]:


res.shape
res.dropna(inplace = True)


# In[145]:


y = res['rent_freq']
x = res.iloc[:,1:]


# In[146]:


x['datetime'] = x['datetime'].astype('category').cat.codes
x['icon'] = x['icon'].astype('category').cat.codes
x['holiday'] = x['holiday'].astype('category').cat.codes
x['weekday'] = x['weekday'].astype('category').cat.codes



x.dtypes


# ## Split the data

# In[147]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25)

#log y label
y_test_Log = np.log1p(y_test)
y_train_Log = np.log1p(y_train)


# ## Random Forest Regression

# In[148]:


def RMSLE(y,ypred):
    y=np.nan_to_num(y)
    ypred=np.nan_to_num(ypred)
    calc=(ypred-y)**2
    return np.sqrt(np.mean(calc))


# In[ ]:





# In[4]:


from sklearn.model_selection import GridSearchCV
import sklearn
rmsle_scorer=sklearn.metrics.make_scorer(RMSLE,greater_is_better=False)

clf_4_cs=RandomForestRegressor()
param={'n_estimators':[200,300,400],'max_depth':[8,9,10]}
grid_4_cs=GridSearchCV(clf_4_cs,param_grid=param,scoring=rmsle_scorer,cv=5,n_jobs=4)
grid_4_cs.fit(x_train, y_train)
print ("Best params",grid_4_cs.best_params_)
print ("RMSLE score for casual train %f" %(RMSLE(y_train, grid_4_cs.predict(x_train))))
print ("RMSLE score for casual test %f" %(RMSLE(y_test, grid_4_cs.predict(x_test))))


# In[3]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

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
  


# In[190]:



#n = 500
score = rfr.score(x_test, y_test)


# In[191]:


score


# ## Gradient Boosting on Regression Tree

# In[ ]:


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


# In[2]:


from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
gbm.fit(x_train,y_train_Log)
preds = gbm.predict(X= x_test)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(y_test_Log),np.exp(preds),False))


# In[ ]:


imshow(confusion_matrix(np.exp(y_test_Log),np.exp(preds)))


# ## SVR

# In[6]:


#find the best
rmsle_scorer=sklearn.metrics.make_scorer(RMSLE,greater_is_better=False)

rg_svr=SVR()
param={'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel' :['rbf','sigmoid','linear']}
grid_4_cs=GridSearchCV(rg_svr,param_grid=param,scoring=rmsle_scorer,cv=5,n_jobs=4)
grid_4_cs.fit(x_train, y_train)
print ("Best params",grid_4_cs.best_params_)
print ("RMSLE score for casual train %f" %(RMSLE(y_train, grid_4_cs.predict(x_train))))
print ("RMSLE score for casual test %f" %(RMSLE(y_test, grid_4_cs.predict(x_test))))


# In[ ]:


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


# In[ ]:


#need modified according to the previous result
from sklearn.svm import SVR
#need to change the kernel from Gaussian Kernel to RBM
svr = SVR()#
y_train_Log = np.log1p(y_train)
svr.fit(x_train,y_train_Log)
y_test_Log = np.log1p(y_test)
preds = svr.predict(X= x_test)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(y_test_Log),np.exp(preds),False))


# In[ ]:


imshow(confusion_matrix(np.exp(y_test_Log),np.exp(preds)))


# In[ ]:


## Cluster

