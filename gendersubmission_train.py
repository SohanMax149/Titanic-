#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


train_data=pd.read_csv(r'C:\ML\Gendersubmission\train.csv')
test_data=pd.read_csv(r'C:\ML\Gendersubmission\test.csv')
gender_submission=pd.read_csv(r'C:\ML\Gendersubmission\gender_submission.csv')


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.describe()


# In[6]:


train_data.dtypes


# In[7]:


train_data.isnull().sum()


# In[8]:


train_data.Survived.value_counts()


# In[9]:


plt=train_data.Survived.value_counts().plot("bar")
plt.set_xlabel("survived or not")
plt.set_ylabel("passenger count")


# In[10]:


plt=train_data.Pclass.value_counts().sort_index().plot("bar", title="")
plt.set_xlabel("pclass")
plt.set_ylabel("survival possibility")


# In[11]:


train_data[['Pclass','Survived']].groupby("Pclass").count()


# In[12]:


train_data[['Pclass','Survived']].groupby("Pclass").sum()


# In[13]:


plt=train_data[['Pclass','Survived']].groupby("Pclass").mean().Survived.plot("bar")
plt.set_xlabel("pclass")
plt.set_ylabel("survival possibility")


# In[14]:


plt=train_data.Sex.value_counts().sort_index().plot("bar")
plt.set_xlabel("Sex")
plt.set_ylabel("Passenger count")


# In[15]:


train_data[['Sex','Survived']].groupby("Sex").count()


# In[16]:


train_data[['Sex','Survived']].groupby("Sex").sum()


# In[17]:


plt=train_data[['Sex','Survived']].groupby("Sex").mean().Survived.plot("bar")
plt.set_xlabel("Sex")
plt.set_ylabel("survival possibility")


# In[18]:


sns.factorplot('Pclass',col= 'Embarked',data= train_data,kind='count')


# In[19]:


sns.factorplot('Sex',col= 'Embarked',data= train_data,kind='count')


# In[20]:


train_data['FamilySize']=train_data['SibSp']+train_data["Parch"]+1


# In[21]:


train_data.head()


# In[22]:


train_data=train_data.drop(columns=['Ticket','PassengerId','Cabin'])


# In[23]:


train_data.head()


# In[24]:


train_data['Sex']=train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked']= train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# In[25]:


train_data.head()


# In[26]:


train_data['Title']= train_data.Name.str.extract('([A-Za-z]+)\.',expand=False)
train_data=train_data.drop(columns ='Name')


# In[27]:


train_data.head()


# In[28]:


train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')


# In[29]:


train_data['Title']=train_data["Title"].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs':3, 'Others': 4})


# In[30]:


train_data.head()


# In[31]:


corr_matrix=train_data.corr()


# In[32]:


import matplotlib.pyplot as plt
plt.figure(figsize=(9,8))
sns.heatmap(data=corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)


# In[33]:


train_data['Embarked']=train_data['Embarked'].fillna(2)


# In[34]:


train_data.head()


# In[35]:


age_median_train=train_data.Age.median()
train_data.Age=train_data.Age.fillna(age_median_train)
print(age_median_train)


# In[36]:


train_data.head()


# In[37]:


from sklearn.utils import shuffle
train_data=shuffle(train_data)


# In[38]:


x_train=train_data.drop(columns='Survived')
y_train=train_data.Survived
y_train=pd.DataFrame({'Survived': y_train.values})


# In[39]:


x_train.shape


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[41]:


x_training, x_valid, y_training, y_valid= train_test_split(x_train, y_train, test_size=0.2)


# In[42]:


logreg_clf=LogisticRegression()
logreg_clf.fit(x_training, y_training)


# In[43]:


prediction=logreg_clf.predict(x_valid)


# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


accuracy_score(y_valid,prediction)


# In[46]:


from sklearn.metrics import confusion_matrix


# In[47]:


confusion=confusion_matrix(y_valid,prediction,labels=[1,0])
print(confusion)


# In[48]:


from sklearn.metrics import classification_report
report=classification_report(y_valid,prediction)
print(report)


# In[49]:


test_data.head()


# In[50]:


test_data.describe()


# In[51]:


test_data.dtypes


# In[52]:


test_data.isnull().sum()


# In[53]:


test_data['FamilySize']=test_data['SibSp']+test_data["Parch"]+1


# In[54]:


test_data.head()


# In[55]:


test_data['Sex']=test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked']=test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# In[56]:


test_data.head()


# In[57]:


test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.', expand=False)
test_data=test_data.drop(columns='Name')


# In[58]:


test_data.head()


# In[59]:


test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')


# In[60]:


test_data['Title']= test_data['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Others': 4})


# In[61]:


test_data.head()


# In[62]:


corr_matrix=test_data.corr()


# In[63]:


plt.figure(figsize=(9,8))
sns.heatmap(data=corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)


# In[64]:


test_data['Fare']=test_data['Fare'].fillna(1)


# In[65]:


test_data.head()


# In[66]:


age_median_test=test_data.Age.median()
test_data.Age=test_data.Age.fillna(age_median_test)
print(age_median_test)


# In[67]:


test_data=test_data.drop(columns=['Ticket','PassengerId','Cabin'])


# In[68]:


test_data.head()


# In[69]:


print(test_data.isnull().sum())


# In[70]:


test_data['Title']=test_data['Title'].fillna(1)


# In[71]:


test_data.head()


# In[72]:


print(test_data.isnull().sum())


# In[73]:


gender_submission['prediction']=logreg_clf.predict(test_data)


# In[74]:


print(gender_submission['Survived'].dtype)


# In[75]:


test_data.head()


# In[76]:


print('Accuracy Score=', accuracy_score(gender_submission['Survived'],gender_submission['prediction']))


# In[77]:


confusion1=confusion_matrix(gender_submission['Survived'],gender_submission['prediction'],labels=[1,0])


# In[78]:


print('Confusion Matrix:',confusion1)


# In[79]:


report1=classification_report(gender_submission['Survived'],gender_submission['prediction'])
print('Report:\n', report1)


# In[ ]:




