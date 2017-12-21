import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Options
learning_rate = 0.001
total_epoch = 30
batch_size = 128
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 참고: BasicLSTMCell, GRUCell
cell = tf.nn.rnn_cell.BasicRnnCell(n_hidden)
# RNN 신경망 구성
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

