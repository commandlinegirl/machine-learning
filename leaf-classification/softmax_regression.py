# Initial version
# TODO: separate out validation data from training

from __future__ import print_function

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

LEARNING_RATE = 0.01
TRAINING_EPOCHS = 2000
NUM_LABELS = 99

# Import train data

train = pd.read_csv('./input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
y_col = train['species']
le = LabelEncoder().fit(y_col)
y_train = le.transform(y_col)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

# Import test data

test = pd.read_csv('./input/test.csv')
test_ids = test['id'].values
x_test = test.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)

# TensorFlow graph input

x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
y_ = tf.placeholder(tf.int64, [None])

W = tf.Variable(tf.zeros([x_train.shape[1], NUM_LABELS]))
b = tf.Variable(tf.zeros([NUM_LABELS]))

# Construct the model

# Compute sparse softmax cross entropy between logits (y) and labels (y_)
# ('sparse' version of the function removed the need to use a one-hot version of labels)
# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

# Minimize the loss function using gradient descent
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Compute the graph in the session
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        # Run optimization op (backprop) and cost op (to get loss value)
        _, loss_value = sess.run([optimizer, cross_entropy], feed_dict={x: x_train,
                                                                        y_: y_train})
        # Display loss value per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(loss_value))

    print("Training finished!")

    # Evaluate the model
    correct_prediction = tf.equal(y_, tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %.9f" % sess.run(accuracy, feed_dict={x: x_test,
                                                           y_: test_ids}))


