#!/usr/bin/env python
# coding: utf-8
# Note tf.multiply is equivalent to "*"
# print((tf.reduce_sum(tf.expand_dims(a,-1) * w, axis=0)).eval())
# print((tf.reduce_sum(a * tf.transpose(w), axis=1)).eval())
import tensorflow as tf
import numpy as np

a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

# <tf.Tensor 'Mul:0' shape=<unknown> dtype=float32>
y = tf.multiply(a, b) # multiply the symbolic variables

# <tf.Tensor 'Placeholder_2:0' shape=(?, 20, 7) dtype=float32>
X=tf.placeholder(tf.float32, shape=[None,20,7]) 

a1 = np.array([[1, 2], [4, 5], [7, 8]])

# print statement have been deprecated,instead of print() # SyntaxError: invalid syntax
# print np.mean(a1, axis=1, keepdims=True)

print("a1_mean= ", np.mean(a1, axis=1, keepdims=True)) 
# http://www.jianshu.com/p/f9e3fd264932
# print np.mean(X, axis=1, keepdims=True)
# axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；
# axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。
# 还可以这么理解，axis是几，那就表明哪一维度被压缩成1。
 #                 [[ 1.5]
 # [[ 4.  5.]]      [ 4.5]    
 #                  [ 7.5]]
with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    # print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
	# print("sess.run(node3): ",sess.run(X))
	# print("X: ",X)
    print("a1= ",a1)
