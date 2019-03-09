# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange
from data_generation import generate_samples

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
H1 = 512 # number of hidden units. 
H2 = 256
H3 = 128 
#H4 = 256
#H5 = 128
H4 = 64

C1D = 256 # filter size
NC1 = 64 # number of channels
C2D = 128# filter size
NC2 = 32 # number of channels
C3D = 64 # filter size
NC3 = 16 # number of channels

P = 4 # number of max pooling * pooling window size

lr = .0001 # the learning rate (previously refered to in the notes as alpha)

#weights and initialization 

X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1],stddev=0.001))
W2 = tf.Variable(tf.truncated_normal([C2D,1,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2], stddev=0.001))
W3 = tf.Variable(tf.truncated_normal([C3D,1,NC2,NC3], stddev=0.001))
b3 = tf.Variable(tf.truncated_normal([NC3], stddev=0.001))

# Fully Connected feed-forward
W_h1 = tf.Variable(tf.truncated_normal([int((D/P)*NC3 * .5),H1], stddev = 0.01)) # mean=0.0
W_h2 = tf.Variable(tf.truncated_normal([H1,H2], stddev = 0.01)) # mean=0.0
W_h3 = tf.Variable(tf.truncated_normal([H2,H3], stddev = 0.01)) # mean=0.0
W_h4 = tf.Variable(tf.truncated_normal([H3,H4], stddev = 0.01)) # mean=0.0
W_o = tf.Variable(tf.truncated_normal([H4,C], stddev = 0.01)) # mean=0.0

b_h1 = tf.Variable(tf.zeros((1, H1)))
b_h2 = tf.Variable(tf.zeros((1, H2)))
b_h3 = tf.Variable(tf.zeros((1, H3)))
b_h4 = tf.Variable(tf.zeros((1, H4)))
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

# Convolution 3

C3_out = tf.nn.conv2d(C2_out_mp, W3, [1,1,1,1], padding='SAME')                                  
C3_out += b3
C3_out = tf.nn.relu(C3_out)  

# Max Pooling 3
C3_out_mp = tf.nn.max_pool(C3_out, ksize = [1,2,1,1], strides = [1,2,1,1], padding='SAME')        

# Flatten
C3_out_mp = tf.reshape(C3_out_mp,[-1, int((D/P)*NC3 * .5)])  

h1 = tf.nn.relu(tf.matmul(C3_out_mp,W_h1) + b_h1)
h2 = tf.nn.relu(tf.matmul(h1,W_h2) + b_h2)
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
h4 = tf.nn.relu(tf.matmul(h3,W_h4) + b_h4)
y_hat = tf.matmul(h4, W_o) + b_o

l = tf.square(tf.add(y_hat, -Y))
loss = tf.reduce_mean(l)

vector_loss = tf.reduce_mean(l,0)

#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(lr, global_step, 50, .96, staircase=True)
GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

data, labels = generate_samples(100)
m, n = data.shape
data = data.reshape([m,n,1,1])

curr_loss = sess.run(loss, feed_dict={X: data, Y: labels})
print ("The initial loss is: ", curr_loss)

x_te, y_te = generate_samples(100)
m, n = x_te.shape
x_te = x_te.reshape([m,n,1,1])
while True:
	x_tr, y_tr = generate_samples(100)
	m, n = x_tr.shape
	x_tr = x_tr.reshape([m,n,1,1])

        for i in range(1000):
	    sess.run(GD_step, feed_dict={X: x_tr, Y: y_tr})

	cur_loss, v_loss = sess.run([loss, vector_loss], feed_dict={X:x_te, Y:y_te})
	print("The vectorized loss is: ", v_loss)
	
	print ("The final training loss is: ", cur_loss)
sess.close()
