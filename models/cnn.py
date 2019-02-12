# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange

# general parameters
D = 8000 # dimensionality of the data
C = 36 # number of unique labels in the dataset

def equation(x):
	return 440 * (2 ** (1/float(12))) ** (x - 49)

#mapping from frequencies to indices 

# mapping = {}
# for i in range(1, 89):
# 	mapping[math.floor(equation(i))] = i

# for i in range(len(y_tr)):
# 	y_tr[i][0] = mapping[math.floor(y_tr[i][0])]

# hyperparameters
H1 = 2048 # number of hidden units. 
H2 = 1024
H3 = 512
H4 = 256
H5 = 128
H6 = 64

C1D = 1024 # filter size
NC1 = 256 # number of channels
C2D = 512 # filter size
NC2 = 64 # number of channels

P = 4 # number of max pooling * pooling window size

lr = .000001 # the learning rate (previously refered to in the notes as alpha)

#weights and initialization 

X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1],stddev=0.001))
W2 = tf.Variable(tf.truncated_normal([C2D,1,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2], stddev=0.001))

# Fully Connected feed-forward
W_h1 = tf.Variable(tf.truncated_normal([int((D/P)*NC2),H1], stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.truncated_normal([H1,H2], stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.truncated_normal([H2,H3], stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.truncated_normal([H3,C], stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_o = tf.Variable(tf.zeros((1, C)))

# Convolution 1

C1_out = tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME')                 
C1_out += b1
C1_out = tf.nn.relu(C1_out)   

C1_out_mp = tf.nn.max_pool(C1_out, ksize = [1,2,1,1], strides=[1,2,1,1], padding='SAME')

# Convolution 2

C2_out = tf.nn.conv2d(C1_out_mp, W2, [1,1,1,1], padding='SAME')                                  
C2_out += b2
C2_out = tf.nn.relu(C2_out)  

# Max Pooling 2
C2_out_mp = tf.nn.max_pool(C2_out, ksize = [1,2,1,1], strides = [1,2,1,1], padding='SAME')        

# Flatten
C2_out_mp = tf.reshape(C2_out_mp,[-1, int((D/P)*NC2)])  

h1 = tf.nn.relu(tf.matmul(C2_out_mp,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
d1 = tf.nn.dropout(h2, .3)
h3 = tf.nn.relu(tf.matmul(d1,W_h3) + b_h3)
y_hat = tf.matmul(h3, W_o) + b_o

l = tf.square(tf.add(y_hat, -Y))
loss = tf.reduce_mean(tf.add(l, l))

vector_loss = tf.reduce_mean(tf.add(l, l), 0)

learning_rate = tf.train.exponential_decay(.01, global_step, 100, .96, staircase=True)
GD_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

data, labels = generate_samples(100)

curr_loss = sess.run(loss, feed_dict={X: data, y: labels})
print ("The initial loss is: ", curr_loss)

x_te, y_te = generate_samples(100)

nepochs = 100
for i in trange(nepochs):
	x_tr, y_tr = generate_samples(100)
	m, n = x_tr.shape
	x_tr = x_tr.reshape([m,n,1,1])

	sess.run(GD_step, feed_dict={X: x_tr, y: y_tr})

	cur_loss, v_loss = sess.run([loss, vector_loss], feed_dict={X:x_te, y:y_te})
	print()
	print("The vectorized loss is: ", v_loss)
	
	print ("The final training loss is: ", cur_loss)

                 
sess.close()
