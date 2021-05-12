#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Importing the datasets
datasetNorm = pd.read_csv('dataset normal.csv')
datasetAttack1 = pd.read_csv('dataset attack1.csv')
datasetAttack2 = pd.read_csv('dataset attack2.csv')

# Add column to label whether it is normal or attack packet
datasetAttack1['Class'] = np.where(datasetAttack1['NDP Message']== 'Router Advertisement', 1, 2)
datasetAttack2['Class'] = np.where(datasetAttack2['NDP Message']== 'Router Advertisement', 1, 2)
datasetNorm['Class'] = '2'

# Merge all datasets together
datasetAll = pd.concat([datasetAttack1, datasetAttack2, datasetNorm])
datasetAll.shape


# In[35]:


# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
nominal_cat = ["MAC Source",  "MAC Destination", "Source", "Destination", "Protocol", "Length", "NDP Message"]
for column in nominal_cat:
    datasetAll[column] = labelencoder.fit_transform(datasetAll[column].astype(str))
datasetAll.head(8)


# In[36]:


# Change data type of 'Class' to integer
datasetAll.Class = datasetAll.Class.astype(int)
datasetAll.dtypes


# In[37]:


# Setting X and Y axis columns
X = datasetAll.drop("Class", axis=1)
Y = datasetAll["Class"]


# In[38]:


# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting the classifier into the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)

# Testing the model by classifying the test set
y_pred = classifier.predict(X_Test)

# Creating confusion matrix for evaluation
cm = confusion_matrix(Y_Test, y_pred)
cr = classification_report(Y_Test, y_pred)

# Print out confusion matrix and report
print(y_pred)
print(cm)
print(cr)


# In[ ]:




