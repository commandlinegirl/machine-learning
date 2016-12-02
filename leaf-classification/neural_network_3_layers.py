# This script has been released under the Apache 2.0 open source license.
# Forked from: https://www.kaggle.com/bmetka/leaf-classification/logistic-regression/run/347146

from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

import tensorflow as tf
from tensorflow.contrib import learn

NUM_CLASSES = 99

# Read in train data
train = pd.read_csv('./input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
y_col = train['species']
le = LabelEncoder().fit(y_col)
y_train = le.transform(y_col)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

# Read in test data
test = pd.read_csv('./input/test.csv')
test_ids = test['id']
x_test = test.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)

print('Number of features:', x_train.shape[1])
print ('Number of train examples:', x_train.shape[0])

# Build 3 layer DNN
classifier = learn.DNNClassifier(
    hidden_units=[384, 227, 193],
    n_classes=NUM_CLASSES,
    feature_columns=learn.infer_real_valued_columns_from_input(x_train),
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.02))

# Fit the model
classifier.fit(x_train, y_train, batch_size=128, steps=10000)

# Make prediction for test data
y_test = classifier.predict_proba(x_test)

# Prepare csv for submission
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission_nn_3_layers.csv')

