import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
import matplotlib.pyplot as plt

### 1. IMPORT DATA
train = pd.read_csv('train.csv', header=0)

# percentage of women in each class
#print 'Percentage of women in each class'
#for i in range(1,4):
#    print i, len(train[ (train['Sex'] == 'female') & (train['Pclass'] == i) ])/float(len(train[train['Pclass'] == i]))


### 2. CLEAN AND FILL IN DATA
# Add Gender column with sex represented numerically
train['Gender'] = train['Sex'].map({'male':0, 'female':1})

# Add PortNum column with embarkation port represented numerically
train['PortNum'] = train['Embarked'].map({ 'S': 0, 'C': 1, 'Q': 2, 'nan':3 })
train.loc[ (train.PortNum.isnull())] = 3

# Fill in age. Use median ages by class and gender
train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
median_ages = train.groupby(['Pclass', 'Gender']).median()
median_ages_age = median_ages['Age']
train['AgeFill'] = train['Age']

# Fill in fare
train['FareFill'] = train['Fare']
median_fares = train.groupby('Pclass')['Fare'].median()
for i in range(1, 4): 
        train.loc[ (train.Fare.isnull()) & (train.Pclass == i), 'FareFill'] = median_fares[i]
          
for i in range(1, 4):
    for j in range(0, 2):
        train.loc[ (train.Age.isnull()) & (train.Pclass == i) & (train.Gender == j), 'AgeFill'] = median_ages_age[i][j]

# Family size
train['FamilySize'] = train['SibSp'] + train['Parch']
train['Age*Class'] = train.AgeFill * train.Pclass

# Uncomment to show object data types
# train.dtypes[train.dtypes.map(lambda x: x=='object')]

# Drop not used columns
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId', 'Fare'], axis=1) 
# (712 rows still contain at least one NA value)

# Print type info
print 'Cleaned training data'
print train.info()

# Convert into data set for skilearn
train_data = train.values




