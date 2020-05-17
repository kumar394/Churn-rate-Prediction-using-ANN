#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# In[2]:


#working directory
os.chdir("C:\\Users\\shive\\Desktop\\MS-BAIM\\Courses\\online courses\\A-Z Deep Learning\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Section 4 - Building an ANN\\Artificial_Neural_Networks")


# # Data Preprocessing

# In[3]:


dataset= pd.read_csv("Churn_Modelling.csv")
dataset.head(5)


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset['Exited'].value_counts()


# In[6]:


#creating dummy variables
geography= pd.get_dummies(dataset["Geography"], drop_first= True)
#geography.head(5)
gender= pd.get_dummies(dataset["Gender"], drop_first= True)


# In[7]:


#dropping irrelevant columns and concatinating created dummy variables
dataset= dataset.drop(["Geography", "Gender", "RowNumber", "CustomerId", "Surname"], axis=1)
dataset= pd.concat([geography, dataset], axis=1)
dataset= pd.concat([gender, dataset], axis=1)
dataset.head(5)


# In[8]:


len(dataset.columns)


# In[9]:


#filtering parameters which might affect churn rate based on business intuition
x= dataset.iloc[:, :11].values
y= dataset.iloc[:, 11].values


# In[10]:


x[:5]


# In[11]:


y[:5]


# In[12]:


#Splitting the dataset into training ans test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.2, random_state= 0)


# In[13]:


x_train[:5]


# In[14]:


y_train[:5]


# In[15]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


# In[16]:


x_train[:5], x_test[:5]


# # Building ANN Model

# In[57]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[18]:


#initializing the ANN
ann_classifier= Sequential()
#adding input layers and hidden layers
ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform", input_shape= (11,)))
ann_classifier.add(Dropout(p= 0.1))
#adding second hidden layers
ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform"))
ann_classifier.add(Dropout(p= 0.1))
#adding the output layer
ann_classifier.add(Dense(1, activation= "sigmoid", kernel_initializer= "uniform"))
#compiling the ANN
ann_classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics= ["accuracy"])


# In[19]:


#fitting the ANN to the training dataset
ann_classifier.fit(x= x_train, y= y_train, batch_size= 10, epochs= 100)


# In[20]:


#predicting the test set result
y_pred= ann_classifier.predict(x_test)
y_pred[:5]


# In[21]:


y_pred= (y_pred>0.5)#considering threshold = 0.5
y_pred[:5]


# In[22]:


#confusion metrix
from sklearn.metrics import confusion_matrix, classification_report
ann_cm= confusion_matrix(y_test, y_pred)
ann_cm


# In[23]:


print(classification_report(y_test, y_pred))


# In[24]:


ann_accuracy= sklearn.metrics.accuracy_score(y_test, y_pred).round(3)
print(ann_accuracy)


# In[25]:


#fitting logistic regression
from sklearn.linear_model import LogisticRegression
lr_classifier= LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)


# In[26]:


#predicting
lr_y_pred= lr_classifier.predict(x_test)
lr_y_pred[:5]


# In[27]:


print(classification_report(y_test, lr_y_pred))


# In[28]:


lr_accuracy= sklearn.metrics.accuracy_score(y_test, lr_y_pred).round(3)
print(lr_accuracy)


# In[29]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
models = ['Logistic Regression', 'Artificial Neural Network']
accuracies = [0.811,0.859]
plt.bar(models, accuracies, width= 0.5)
plt.title("Accuracy Comparison")
plt.ylabel("accuracy")
plt.yticks(np.arange(0,1.1,0.1))
plt.show()


# In[31]:


"""Prediction for customer with following attributes
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""
new_prediction = ann_classifier.predict(sc.transform(np.array([[0.0, 0, 0, 600, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction= (new_prediction > 0.5)
print(new_prediction)


# # Evaluating, Improving and Tuning the ANN

# In[45]:


#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_ann_classifier():
    ann_classifier= Sequential()
    ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform", input_shape= (11,)))
    ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform"))
    ann_classifier.add(Dense(1, activation= "sigmoid", kernel_initializer= "uniform"))
    ann_classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics= ["accuracy"])
    return ann_classifier


# In[46]:


ann_classifier= KerasClassifier(build_fn= build_ann_classifier, batch_size= 10, nb_epoch= 100)


# In[47]:


accuracies= cross_val_score(estimator= ann_classifier, X= x_train, y= y_train, cv=10, n_jobs= -1)


# In[56]:


mean= round(accuracies.mean(),3)
variance= round(accuracies.std(),3)
print("Mean Accuracy:", mean)
print("Variance:", variance)


# In[65]:


#tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_ann_classifier(optimizer):
    ann_classifier= Sequential()
    ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform", input_shape= (11,)))
    ann_classifier.add(Dense(6, activation= "relu", kernel_initializer= "uniform"))
    ann_classifier.add(Dense(1, activation= "sigmoid", kernel_initializer= "uniform"))
    ann_classifier.compile(optimizer=optimizer, loss= "binary_crossentropy", metrics= ["accuracy"])
    return ann_classifier

ann_classifier= KerasClassifier(build_fn= build_ann_classifier)
parameters= {"batch_size": [10, 25, 32],
            "nb_epoch": [50, 100, 500],
            "optimizer": ["adam", "rmsprop"]}
grid_search= GridSearchCV(estimator= ann_classifier, 
                          param_grid= parameters, 
                          cv= 10,  
                          scoring= 'accuracy')
grid_search= grid_search.fit(x_train, y_train)
best_parameter= grid_search.best_params_
best_accuracy= grid_search.best_score_


# In[66]:


best_accuracy


# In[67]:


best_parameter


# In[ ]:




