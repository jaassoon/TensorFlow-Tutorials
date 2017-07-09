#!/usr/bin/env python
import tensorflow as tf
import numpy as np
# A tensor's rank is its number of dimensions. tensor维度
# 3 # a rank 0 tensor; this is a scalar with shape [] #just value, no rank, rank=0
# [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
# rank=1
# 它只有在第一個矩陣的列數（column）和第二個矩陣的行數（row）相同時才有定義。
# The matmul operation only works on matrices (2D tensors).
vec = np.array([
			  20, 50
			 ])
# shape(2,2)
a = np.array([
			  [20, 5],
			  [15, 10]
			 ])
# shape(2,2)
w = np.array([[2.0, 1.0], 
			  [1.0, 4.0]
			 ])

a = tf.constant(a, dtype=tf.float64)
vec = tf.constant(vec, dtype=tf.float64)
w = tf.constant(w)

with tf.Session() as sess:
  # reference:[Tensorflow一些常用基本概念与函数（1）](http://blog.csdn.net/lenbow/article/details/52152766)
  print(sess.run(tf.rank(vec))) # 1
  print(tf.matmul(a, w).eval())	 # [[ 45.  40.][ 40.  55.]]
  print(tf.matmul(tf.expand_dims(vec,0), w).eval()) # [[  90.  220.]]

  # print(tf.matmul(tf.expand_dims(a,-1), w).eval()) # ValueError: Shape must be rank 2 but is rank 3 for 'MatMul' (op: 'MatMul') with input shapes: [2,2,1], [2,2].
