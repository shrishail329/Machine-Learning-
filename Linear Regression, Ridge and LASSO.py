#!/usr/bin/env python
# coding: utf-8

# # Welcome to Regression Lession 

# ## Linear Regression, Logistic and LASSO

# In[1]:


from sklearn.datasets import load_boston


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = load_boston()


# In[6]:


type(df)


# In[8]:


df


# In[12]:


pd.DataFrame(df.data)


# In[15]:


dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()


# In[32]:


dataset['price'] = df.target


# In[34]:


dataset.head()


# In[35]:


## Dividing the dataset into independent and dependent features
X = dataset.iloc[:,:-1] ## Independent feature 
y = dataset.iloc[:,-1] ## Dependent feature 


# In[36]:


X.head()


# In[37]:


y.head()


# In[40]:


## Linear Regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score # cross validation is basically take first cross validation data that will be test data
                                                   # and remaining will train data 
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, X, y,scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
print(mean_mse)


# In[48]:


# Ridge Regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[53]:


# Hyper parametric tuning 
param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20]}
ridge = Ridge()

ridge_reggressor = GridSearchCV(ridge,param,scoring='neg_mean_squared_error',cv=5)
ridge_reggressor.fit(X,y)


# In[54]:


print(ridge_reggressor.best_params_)
print(ridge_reggressor.best_score_)


# In[65]:


## LASSO Regression 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20]}
lasso_reggressor = GridSearchCV(lasso,param,scoring='neg_mean_squared_error',cv=5)
lasso_reggressor.fit(X,y)


# In[66]:


print(lasso_reggressor.best_params_)
print(lasso_reggressor.best_score_)


# In[68]:


# Hyper parametric tuning 
param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}
ridge = Ridge()

ridge_reggressor = GridSearchCV(ridge,param,scoring='neg_mean_squared_error',cv=5)
ridge_reggressor.fit(X,y)


# In[69]:


print(ridge_reggressor.best_params_)


# In[70]:


print(ridge_reggressor.best_score_)


# In[71]:


## LASSO Regression 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}
lasso_reggressor = GridSearchCV(lasso,param,scoring='neg_mean_squared_error',cv=5)
lasso_reggressor.fit(X,y)


# In[72]:


print(lasso_reggressor.best_params_)
print(lasso_reggressor.best_score_)


# ##### we can observe here after increasing the feature the lasso is not increasing but ridge is incresed 

# ### Train Test split 

# In[81]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42) # here we are selecting 33 % data for test and 77% data for training 

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
mse = cross_val_score(lin_reg, X_train, y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
print(mean_mse)


# In[75]:


# Hyper parametric tuning 
# Ridge Regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}
ridge = Ridge()

ridge_reggressor = GridSearchCV(ridge,param,scoring='neg_mean_squared_error',cv=5)
ridge_reggressor.fit(X_train,y_train)


# In[76]:


print(ridge_reggressor.best_params_)
print(ridge_reggressor.best_score_)


# In[77]:


## LASSO Regression 
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
param = {'alpha':[1e-15,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}
lasso_reggressor = GridSearchCV(lasso,param,scoring='neg_mean_squared_error',cv=5)
lasso_reggressor.fit(X_train,y_train)


# In[78]:


print(lasso_reggressor.best_params_)
print(lasso_reggressor.best_score_)


# In[79]:


y_pred = lasso_reggressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred,y_test)


# In[80]:


print(r2_score1)


# In[ ]:


from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred,y_test)


# In[82]:


y_pred = ridge_reggressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)


# In[83]:


y_pred = lin_reg.predict(X_test)
from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)


# ## Logistic Regression  

# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


# In[88]:


df = load_breast_cancer()
## Independent features
pd.DataFrame(df.data)


# In[92]:


## Independenet feature 
X = pd.DataFrame(df['data'],columns=df['feature_names'])


# In[93]:


X.head()


# In[95]:


# Dependent feature 
y = pd.DataFrame(df['target'],columns=['Target'])
y


# In[97]:


y['Target'].value_counts() 


# In[98]:


## Train test split 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[100]:


params = [{'C':[1,5,10]},{'max_iter':[100,150]}]


# In[101]:


model1 = LogisticRegression(C = 100,max_iter=100)


# In[102]:


model1 = GridSearchCV(model1,param_grid=params,scoring='f1',cv = 5)


# In[103]:


model1.fit(X_train,y_train)


# In[104]:


model1.best_params_


# In[105]:


model1.best_score_


# In[114]:


y_pred = model1.predict(X_test)


# In[115]:


y_pred


# In[116]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[117]:


confusion_matrix(y_test,y_pred)


# In[118]:


accuracy_score(y_test,y_pred)


# In[120]:


print(classification_report(y_test,y_pred))


# In[ ]:




