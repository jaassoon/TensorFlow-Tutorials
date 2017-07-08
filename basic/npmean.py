#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

# <tf.Tensor 'Placeholder_2:0' shape=(?) dtype=float32>
X0=tf.placeholder(tf.float32) 
X1=tf.placeholder(tf.float32) 
# shape(3,2)
a1 = np.array([[1, 2], [4, 5], [7, 8]])

# shape(1,2)
axis0= np.mean(a1, axis=0, keepdims=True)
# shape(3,1)
axis1= np.mean(a1, axis=1, keepdims=True)
# http://www.jianshu.com/p/f9e3fd264932
# axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；
# axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。
# 还可以这么理解，axis是几，那就表明哪一维度被压缩成1。
#                 [[ 1.5]
# [[ 4.  5.]]      [ 4.5]    
#                  [ 7.5]]
with tf.Session() as sess: # create a session to evaluate the symbolic expressions
	print(sess.run(X0, feed_dict={X0: axis0}))
	print(sess.run(X1, feed_dict={X1: axis1}))
	# shape(3,2)
	print(sess.run(X1*X0, feed_dict={X0: axis0,X1: axis1}))
