#!/usr/bin/env python
# coding: utf-8

# In[228]:


import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score 


# In[229]:


data = pd.read_csv("C:\\Users\\monty\\OneDrive\\Desktop\\intenship\\Titanic-Dataset.csv")


# In[230]:


data.head(5)


# In[231]:


data.info()


# In[232]:


data.isnull().sum()


# In[233]:


#removing cabin unnecessary coloumn 
data= data.drop(columns= 'Cabin',axis=1)


# In[234]:


data.isnull().sum()


# #replacing age(column) null value  with mean of age column
# 

# In[235]:


data["Age"].mean()


# In[236]:


data["Age"].fillna(data['Age'].mean(),inplace=True)


# In[237]:


data.isnull().sum()


# In[238]:


data.info()


# # Fixing embarked column 
# 

# In[239]:


print(data["Embarked"].mode())


# In[240]:


print(data["Embarked"].mode()[0])


# In[241]:


#replacing the nulll value of Embarked column. with its mode 
data["Embarked"].fillna(data["Embarked"].mode()[0],inplace=True)


# In[242]:


data.isnull().sum()


# # analysis the data 

# In[243]:


data.describe()


# In[244]:


#how many servived?
#0=non survived  && 1=survived
data['Survived'].value_counts()


# # visuallization 
# 

# In[245]:


sns.set()


# In[246]:


sns.countplot(data['Survived'])


# In[247]:


data["Sex"].value_counts()


# In[248]:


sns.countplot(x='Sex', data=data)


# In[249]:


sns.countplot(x='Sex', hue = 'Survived' ,data=data)


# In[250]:


sns.countplot(x='Pclass',data=data)


# In[251]:


data['Sex'].value_counts()


# In[252]:


data['Embarked'].value_counts()


# In[253]:


data.replace({'Sex':{'male':1,'female':0}, 'Embarked':{'S':0 ,'C':1, 'Q':2}})


# In[254]:


X=data.drop(columns= ['PassengerId','Name','Ticket','Survived'], axis=1)
Y=data['Survived']


# In[255]:


print(X
     )


# In[256]:


print(Y)


# #spliting the data into test and train data 
# 

# In[257]:


X_train, X_valid, y_train, y_valid = train_test_split(X,Y, test_size=0.2, random_state=7)


# In[258]:


print(X.shape, X_train.shape,X_valid.shape)


# # logestical regression and model training

# In[259]:


model = LogisticRegression()


# In[260]:


from sklearn.tree import DecisionTreeClassifier


# In[261]:


dtc=DecisionTreeClassifier()


# In[265]:


import pandas as pd


# Read the dataset
data = pd.read_csv("C:\\Users\\monty\\OneDrive\\Desktop\\intenship\\Titanic-Dataset.csv")

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data)

# Split data into features (X) and target variable (y)
X = data_encoded.drop(['Age', 'Survived'], axis=1)
y = data_encoded['Survived']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)



# In[267]:


# Assuming you have already imported LogisticRegression and split your data into X_train, X_test, y_train, and y_test

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Now that the model is trained, you can make predictions
X_train_prediction = model.predict(X_train)


# In[268]:


print(X_train_prediction)


# In[271]:


traning_data_accuracy=accuracy_score(y_train,X_train_prediction)


# In[272]:


print('Accuracy score of training data :',traning_data_accuracy)


# In[ ]:


#check the accuracy of the test data


# In[275]:


X_test_prediction= model.predict (X_test)


# In[283]:


print(X_test_prediction)


# In[279]:


test_data_accuracy= accuracy_score(y_test,X_test_prediction)


# In[280]:


print('Accuracy score of the test data :',test_data_accuracy)


# In[ ]:


#ends but


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




