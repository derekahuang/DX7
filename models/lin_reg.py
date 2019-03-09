# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange
from data_generation import generate_samples

# data = np.load('../samples.npy')
# labels = np.load('../labels.npy')


# general parameters
D = 8000 # dimensionality of the data
C = 36

def equation(x):
	return 440 * (2 ** (1/float(12))) ** (x - 49)

#mapping from frequencies to indices 

# mapping = {}
# for i in range(1, 89):
# 	mapping[math.floor(equation(i))] = i

# for i in range(len(y_tr)):
# 	y_tr[i][0] = mapping[math.floor(y_tr[i][0])]

# hyperparameters
H1 = 1024 # number of hidden units. In general try to stick to a power of 2
H2 = 512
H3 = 128
lr = .0001 # the learning rate (previously refered to in the notes as alpha)

W_h1 = tf.Variable(tf.random_normal((D,H1), stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.random_normal((H1,H2), stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.random_normal((H2,H3), stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.random_normal((H3,C), stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_o = tf.Variable(tf.zeros((1, C)))

X = tf.placeholder("float", shape=[None,D])
y = tf.placeholder("float", shape=[None,C])

h1 = tf.nn.relu(tf.matmul(X,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
y_hat = tf.matmul(h3, W_o) + b_o

l = tf.square(tf.add(y_hat, -y))
loss = tf.reduce_mean(tf.add(l, l))

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(.01, global_step, 100, .96, staircase=True)

vector_loss = tf.reduce_mean(tf.add(l, l), 0)
GD_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

data, labels = generate_samples(100)

curr_loss = sess.run(loss, feed_dict={X: data, y: labels})
print ("The initial loss is: ", curr_loss)

x_te, y_te = generate_samples(100)

nepochs = 10000
for i in trange(nepochs):
	x_tr, y_tr = generate_samples(100)

	sess.run(GD_step, feed_dict={X: x_te, y: y_te})

	cur_loss, v_loss = sess.run([loss, vector_loss], feed_dict={X:x_te, y:y_te})
	print()
	print("The vectorized loss is: ", v_loss)
	
	print ("The final training loss is: ", cur_loss)

                 
sess.close()
