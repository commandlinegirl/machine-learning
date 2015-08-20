import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
import matplotlib.pyplot as plt

### 1. IMPORT DATA
test = pd.read_csv('test.csv', header=0)

# percentage of women in each class
#print 'Percentage of women in each class'
#for i in range(1,4):
#    print i, len(test[ (test['Sex'] == 'female') & (test['Pclass'] == i) ])/float(len(test[test['Pclass'] == i]))


### 2. CLEAN AND FILL IN DATA
# Add Gender column with sex represented numerically
test['Gender'] = test['Sex'].map({'male':0, 'female':1})

# Add PortNum column with embarkation port represented numerically
test['PortNum'] = test['Embarked'].map({ 'S': 0, 'C': 1, 'Q': 2, 'nan':3 })
test.loc[ (test.PortNum.isnull())] = 3

# Fill in age. Use median ages by class and gender
test['AgeIsNull'] = pd.isnull(test.Age).astype(int)
median_ages = test.groupby(['Pclass', 'Gender']).median()
median_ages_age = median_ages['Age']
test['AgeFill'] = test['Age']

for i in range(1, 4):
    for j in range(0, 2):
        test.loc[ (test.Age.isnull()) & (test.Pclass == i) & (test.Gender == j), 'AgeFill'] = median_ages_age[i][j]

# Fill in fare
test['FareFill'] = test['Fare']
median_fares = test.groupby('Pclass')['Fare'].median()
for i in range(1, 4):
    test.loc[ (test.Fare.isnull()) & (test.Pclass == i), 'FareFill'] = median_fares[i]

# Family size
test['FamilySize'] = test['SibSp'] + test['Parch']
test['Age*Class'] = test.AgeFill * test.Pclass

# Uncomment to show object data types
# test.dtypes[test.dtypes.map(lambda x: x=='object')]
# test.PortNum[~test.PortNum.isin([0, 1, 2])]


# Drop not used columns
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId', 'Fare'], axis=1) 
# (712 rows still contain at least one NA value)

# Print type info
print 'Cleaned testing data'
print test.info()

# Convert into data set for skilearn
test_data = test.values




