
# coding: utf-8

# Exercise from "Learning Tensorflow", O'Reilly

# In[12]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ### Vanilla RNN
# 
# Define the update step
# 
# $$h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)$$

# We will treat each 28x28-pixel MNIST image as a sequence
# of length 28, each sequence is a vector of length 28

# In[3]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/mnist_data", one_hot=True)


# In[4]:

# Define some parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128


# In[5]:

# Path for TensorBoard model summaries
LOG_DIR = "logs/RNN_with_summaries"


# In[6]:

# Placeholders
_inputs = tf.placeholder(tf.float32,
                         shape=[None, time_steps, element_size],
                         name = 'inputs')
y = tf.placeholder(tf.float32,
                   shape=[None, num_classes],
                   name = 'labels')


# In[7]:

batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = batch_x.reshape((batch_size, time_steps, element_size))


# ### Logging helper

# In[10]:

# This helper function taken from official TensorFlow documentation,
# simply add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# ### RNN step

# In[14]:

# Weights and bias for input and hiddel layers
with tf.name_scope('rnn_weights'):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


# In[15]:

def rnn_step(previous_hidden_state, x):
    return tf.tanh(tf.matmul(x, Wx) + 
                   tf.matmul(previous_hidden_state, Wh) + 
                   b_rnn)


# In[18]:

# Transpose _input to shape: time_steps, batch_size, element_size
# in order that we can iterate over time steps with tf.scan
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])


# In[21]:

# Initialize hidden layer
initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# Get all state vectors
all_hidden_states = tf.scan(rnn_step,
                            processed_input,
                            initializer=initial_hidden,
                            name='states')


# ### Sequential outputs

# In[24]:

# Weights for output layers
with tf.name_scope('linear_layer_weights'):
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,
                                              num_classes],
                                              mean=0,
                                              stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope('bias_linear'):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                             mean=0,
                                             stddev=.01))
        variable_summaries(bl)

        
# Apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl


with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # Get last output -- h_28
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)


# ### RNN clasification

# In[28]:

with tf.name_scope('cross_entropy'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    cross_entropy = tf.reduce_mean(loss)
    tf.summary.scalar('cross_entropy', cross_entropy)
    
with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
    
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# #### Merge all summaries

# In[33]:

merged = tf.summary.merge_all()


# ### Run the model

# In[31]:

test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size)) # -1 is used to infer the shape
test_label = mnist.test.labels[:batch_size]


# In[34]:

NUM_STEPS = 10000
with tf.Session() as sess:
    # Write TensorBoard summaries to LOG_DIR
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                        graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    
    for i in range(NUM_STEPS):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequences of 28 pixels
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary, _ = sess.run([merged, train_step],
                              feed_dict={_inputs: batch_x, y: batch_y})
        # Add to summaries
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            acc, loss, = sess.run([accuracy, cross_entropy],
                                  feed_dict={_inputs: batch_x,
                                             y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        if i % 100 == 0:
            # Calculate accuracy for 128 mnist test images and
            # add to summaries
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={_inputs: test_data,
                                               y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: test_label})
    print("Test Accuracy:", test_acc)


# In[ ]:



