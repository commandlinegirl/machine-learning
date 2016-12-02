from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

train = pd.read_csv('./input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
y_col = train['species']
le = LabelEncoder().fit(y_col)
y_train = le.transform(y_col)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

print('Number of features:', x_train.shape[1])
print ('Number of train examples:', x_train.shape[0])

lr = LogisticRegression()
lr.fit(x_train, y_train)

test = pd.read_csv('./input/test.csv')
test_ids = test['id']
x_test = test.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)

y_test = lr.predict_proba(x_test)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission_logistic_regression_simple.csv')

