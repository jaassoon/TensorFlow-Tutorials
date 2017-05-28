import numpy as np
# A tensor's rank is its number of dimensions. 
# 3 # a rank 0 tensor; this is a scalar with shape [] #just value, no rank, rank=0
# [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
# rank=1
vec = np.array([
			  20, 50
			 ])
a = np.array([
			  # 20, 50
			  [20, 5],
			  [15, 10]
			 ])
w = np.array([[2.0, 1.0], 
			  [1.0, 4.0]
			 ])

# print(np.dot(a, w))
# [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
# print(np.sum(np.expand_dims(a, -1) * w , axis=0))
# equivalent result [2.6, 3]

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
# print(node1, node2)

a = tf.constant(a, dtype=tf.float64)
vec = tf.constant(vec, dtype=tf.float64)
w = tf.constant(w)

with tf.Session() as sess:
  # they all produce the same result as numpy above
  # print(tf.constant(vec).eval())
  # print(tf.expand_dims(vec,0))
  print(tf.expand_dims(vec,0).eval())
  # print(tf.matmul(a, w).eval())
  # print(tf.matmul(tf.expand_dims(a,0), w).eval())
  # print(tf.matmul(tf.expand_dims(a,0), w).eval())
  # print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0)).eval())
  # print((tf.reduce_sum(tf.multiply(a, tf.transpose(w)), axis=1)).eval())

  # Note tf.multiply is equivalent to "*"
  # print((tf.reduce_sum(tf.expand_dims(a,-1) * w, axis=0)).eval())
  # print((tf.reduce_sum(a * tf.transpose(w), axis=1)).eval())