#!/usr/bin/env python
# coding: utf-8

# NAME : Prajakta Ramesh Chavan
# Date : 27/02/2024
# TASK_05: Sales Prediction Using Python
# INTERNSHIP : OASIS INFOBYE.

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from colorama import Fore, Back, Style
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold


# In[5]:


df = pd.read_csv(r'C:\Users\Dell\Desktop\Advertising.csv')


# In[6]:


df


# In[7]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.size


# In[10]:


df.columns


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.drop(columns="Unnamed: 0",inplace=True)


# In[14]:


df.head()


# In[15]:


df.duplicated().sum()


# In[16]:


df.isnull().sum()


# In[17]:


df.corr()


# In[18]:


df.corr()["Sales"].sort_values()


# In[24]:


corr=df.corr()
sns.heatmap(corr,annot=True,color='b')
plt.show()


# In[25]:


plt.figure(figsize=[5,3])
plt.boxplot(df,vert=False,data=df,labels=df.columns)
plt.show()


# In[26]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


# In[27]:


index = 1
for feature in df.columns:
    plt.figure(figsize=(30,8))
    
    #first plot
    plt.subplot(index,2,1)
    plt.title(feature+" Distribution Plot")
    sns.distplot(df[feature])
    
    # second plot
    plt.subplot(index,2,2)
    plt.title(feature+" Box Plot")
    sns.boxplot(y=df[feature])
    
    index+=1
    plt.show()


# In[28]:


plt.hist(df['Sales'])
plt.show()


# In[29]:


sns.distplot(df['Sales'],hist=False)
plt.show()


# In[30]:


plt.figure(figsize=(6,6))
sns.distplot(df['Sales'])
plt.title('Sales Distribution')
plt.show()


# In[31]:


plt.figure(figsize=(6,6))
sns.distplot(df['TV'])
plt.title('TV Distribution')
plt.show()


# In[32]:


plt.figure(figsize=(6,6))
sns.distplot(df['Radio'])
plt.title('Radio Distribution')
plt.show()


# In[33]:


plt.figure(figsize=(6,6))
sns.distplot(df['Newspaper'])
plt.title('Newspaper Distribution')
plt.show()


# In[34]:


#Pair plot for all features 
plt.figure(figsize=(8,5),dpi=100)
sns.pairplot(df)


# In[35]:


sns.pairplot(df,hue='TV')


# In[36]:


sns.pairplot(df,hue='Radio')


# In[37]:


sns.pairplot(df,hue='Newspaper')


# In[38]:


sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'],
            y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[39]:


sns.jointplot(x = 'TV',y = 'Sales', data = df, color='blue')


# In[40]:


sns.jointplot(x = 'Radio',y = 'Sales', data = df, color='yellow')


# In[41]:


sns.jointplot(x = 'Newspaper',y = 'Sales', data = df, color='pink')


# In[42]:


sns.histplot(df['TV'])


# In[43]:


sns.histplot(df['Newspaper'])


# In[44]:


sns.histplot(df['Radio'])


# In[45]:


sns.histplot(df['Sales'])


# In[46]:


plt.figure(figsize=(20,7))

plt.subplot(1,3,1)
sns.scatterplot(x='TV', y='Sales', data= df)

plt.subplot(1,3,2)
sns.scatterplot(x='Radio', y='Sales', data= df)

plt.subplot(1,3,3)
sns.scatterplot(x='Newspaper', y='Sales', data= df)
plt.show()


# In[47]:


sns.scatterplot(data = df, x='TV', y='Sales', color = 'y')
plt.title('TV vs Sales')
plt.show()


# In[48]:


sns.scatterplot(data = df, x='Newspaper', y='Sales', color = 'b')
plt.title('Newspaper vs Sales')
plt.show()


# In[49]:


sns.scatterplot(data = df, x='Radio', y='Sales', color = 'green')
plt.title('Radio vs Sales')
plt.show()


# In[50]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['TV'],y=df['Sales'])
plt.show()


# In[51]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Radio'],y=df['Sales'])
plt.show()


# In[52]:


plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Newspaper'],y=df['Sales'])
plt.show()


# In[53]:


X=df.drop('Sales',axis=1)
X.head()


# In[54]:


y=df['Sales']
y.head()


# In[55]:


y.tail()


# In[56]:


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)


# In[57]:


print('Training Data Shape Of X and y : ',X_train.shape,y_train.shape)
print('Testing Data Shape Of X and y  : ',X_test.shape,y_test.shape)


# In[58]:


X_train.tail()


# In[59]:


y_test.tail()


# In[61]:


print("The Shape of X_train dataset :", X_train.shape)
print("The Shape of X_test dataset  :", X_test.shape)
print("The Shape of y_train dataset :", y_train.shape)
print("The Shape of y_test dataset  :", y_test.shape)


# In[62]:


model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print('Linear Regression MSE:', mse_lr)


# In[63]:


model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print('Decision Tree MSE:', mse_dt)


# In[64]:


model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Random Forest MSE:', mse_rf)


# In[66]:


print('Linear Regression R2 :', model_lr.score(X_test, y_test))
print('Decision Tree R2     :', model_dt.score(X_test, y_test))
print('Random Forest R2     :', model_rf.score(X_test, y_test))


# In[70]:


from sklearn import metrics
print('Mean Absolute Error    :',metrics.mean_absolute_error(y_pred_dt,y_test))
print('Root MeanSquare Error  :',np.sqrt(metrics.mean_squared_error(y_pred_dt,y_test)))
print('R-Squared              :',metrics.r2_score(y_pred_dt,y_test))


# In[72]:


from sklearn import metrics
print('Mean Absolute Error      :',metrics.mean_absolute_error(y_pred_lr,y_test))
print('Root MeanSquare Error    :',np.sqrt(metrics.mean_squared_error(y_pred_lr,y_test)))
print('R-Squared                :',metrics.r2_score(y_pred_lr,y_test))


# In[73]:


models = {'dt' : DecisionTreeRegressor(),
          'lr' : LinearRegression(),
          'random' : RandomForestRegressor()}


# In[74]:


report = {}
for i in range(len(list(models))):
            model = list(models.values())[i]
            print(f"Model Training started with {model}")
            model.fit(X_train,y_train)
            print(f"Training completed successfully")
            y_test_pred = model.predict(X_test)
            print("Calculating score")
            test_model_score = r2_score(y_test, y_test_pred)
            print(f"Calculted score: {round(test_model_score*100,2)}% for {model}")
            print("=="*30)
            report[list(models.keys())[i]] = test_model_score


# In[75]:


print("MAE        :", metrics.mean_absolute_error(y_test, y_pred_rf))
print("MSE        :", metrics.mean_squared_error(y_test, y_pred_rf))
print("R2_score   :", metrics.r2_score(y_test, y_pred_rf))


# In[ ]:




