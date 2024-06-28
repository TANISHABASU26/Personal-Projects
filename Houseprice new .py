#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install numpy


# In[4]:


pip install pandas


# In[5]:


pip install matplotlib seaborn scikit-learn


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


data = pd.read_csv("C:\\Users\\tanis\OneDrive\\Desktop\\OFFER LETTERS\\DATA SET USED FOR PROJECTS\\housing.csv")


# In[8]:


data


# In[9]:


data.info()


# In[10]:


data.dropna(inplace = True) #drops the values which not present or missing.


# In[11]:


data.info()


# In[12]:


#training and testing data


# In[13]:


from sklearn.model_selection import train_test_split #define x and y first
x = data.drop(['median_house_value'], axis=1) #axis is = 1 since we are considering a column here
y = data['median_house_value']


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2) #this is done to split the data into two sections, training and test data, by default 0.2 or 20% data is used for training


# In[15]:


train_data = x_train.join(y_train)


# In[16]:


train_data.corr() #correlation among variables


# In[17]:


train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1 )
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1 )
train_data['population'] = np.log(train_data['population'] + 1 )
train_data['households'] = np.log(train_data['households'] + 1 )
train_data.hist(figsize = (15,8))


# In[18]:


train_data


# In[19]:


train_data.ocean_proximity


# In[20]:


pd.get_dummies(train_data.ocean_proximity)


# In[21]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)


# In[22]:


train_data


# In[23]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True , cmap ="YlGnBu")


# In[24]:


plt.figure(figsize = (15,8))
sns.scatterplot(x='latitude' , y = 'longitude', data = train_data , hue = 'median_house_value' , palette = 'coolwarm')


# In[25]:


train_data['bedroom_ratio'] = train_data['total_bedrooms']/train_data['total_rooms']
train_data['households_rooms'] = train_data['total_rooms']/train_data['households']


# In[26]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True , cmap ="YlGnBu")


# In[27]:


from sklearn.linear_model import LinearRegression


x_train,y_train = train_data.drop(['median_house_value'], axis = 1) , train_data['median_house_value']

reg = LinearRegression()
reg.fit(x_train , y_train)


# In[28]:


test_data = x_test.join(y_test)


# In[29]:


test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)


# In[30]:


test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)


# In[31]:


test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['households_rooms'] = test_data['total_rooms'] / test_data['households']


# In[32]:


x_test,y_test = test_data.drop(['median_house_value'], axis = 1) , test_data['median_house_value']


# In[33]:


test_data


# In[35]:


reg.score(x_test,y_test)


# In[36]:


#RANDOM FOREST MODEL


# In[43]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
forest = RandomForestRegressor()

x_train_s = scaler.fit_transform(x_train)

forest.fit(x_train_s, y_train)


# In[47]:


x_test_s = scaler.transform(x_test)


# In[48]:


forest.score(x_test_s,y_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split" : [2, 4],
    "max_depth":[None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(x_train_s, y_train)


# In[ ]:


best_forest = grid_search.best_estimator_


# In[ ]:


best_forest.score(x_test_s, y_test)

