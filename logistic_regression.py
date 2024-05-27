#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing=
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[3]:


disease_df = pd.read_csv('framingham.csv')


# In[4]:


disease_df.head()


# In[5]:


#removing the education column as its irrelevant
disease_df.drop(['education'], inplace=True, axis=1)


# In[6]:


disease_df.isnull().sum()


# In[7]:


disease_df.dropna(axis=0, inplace=True)


# In[8]:


disease_df.shape


# In[9]:


disease_df.TenYearCHD.value_counts()


# In[10]:


#splitting the dataset into Test and train 
X = np.asarray(disease_df[['age','male','cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])


# In[11]:


X = preprocessing.StandardScaler().fit(X).transform(X)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.3, random_state = 4)


# In[13]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[14]:


# Exploratory Data Analysis of Heart Disease Dataset


# In[15]:


plt.figure(figsize=(7,5))
sns.countplot(x='TenYearCHD',data=disease_df)
plt.show()


# In[16]:


lst = disease_df['TenYearCHD'].plot()
plt.show(lst)


# In[17]:


# Fitting Logistic Regression Model for Heart Disease Prediction
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[19]:


# Evaluating Logistic Regression Model
from sklearn.metrics import accuracy_score
print('accuracy_score: ', accuracy_score(y_test,y_pred))


# In[20]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, columns =['Predicted: 0', 'Predicted:1'],
                          index=['Actual:0','Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix,annot = True, fmt = 'd', cmap = "Greens")
plt.show()
print (classification_report(y_test, y_pred))


# In[ ]:




