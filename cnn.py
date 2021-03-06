# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm, trange

data = np.load('samples.npy')
labels = np.load('labels.npy')#[:,0:1]

m,n = data.shape
print(data[0])
data = data.reshape([m,n,1,1])
print(data[0])

indices = random.sample(range(0, 35200), 100)

x_tr = data[indices] #[data[v] for v in indices]
y_tr = labels[indices] #[labels[v] for v in indices]

# general parameters
N = x_tr.shape[0] # number of training examples
D = x_tr.shape[1] # dimensionality of the data
C = y_tr.shape[1] # number of unique labels in the dataset

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
NC2 = 16 # number of channels

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
h3 = tf.nn.relu(tf.matmul(h2,W_h3) + b_h3)
y_hat = tf.matmul(h3, W_o) + b_o

l = tf.square(tf.add(y_hat, -Y))
loss = tf.reduce_mean(tf.add(l, l))

vector_loss = tf.reduce_mean(tf.add(l, l), 0)

GD_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

curr_loss = sess.run(loss, feed_dict={X: x_tr, Y: y_tr})
print ("The initial loss is: ", curr_loss)

sess.run(GD_step, feed_dict={X: x_tr, Y: y_tr})

x_te = data[35100:35200]
y_te = labels[35100:35200]

nepochs = 100
for i in trange(nepochs):
	r = np.random.permutation(35100)
	for j in trange(351):
		indices = r[j*100:(j+1)*100]
		x = data[indices] #[data[v] for v in indices]
		y = labels[indices] #[labels[v] for v in indices]

		sess.run(GD_step, feed_dict={X: x, Y: y})

	cl = sess.run(loss, feed_dict={X:x_te, Y:y_te})
	vl = sess.run(vector_loss, feed_dict={X:x_te, Y:y_te})
	print()
	print("The vectorized loss is: ", vl)
	
	print ("The final training loss is: ", cl)

                 
sess.close()
