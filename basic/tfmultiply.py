#!/usr/bin/env python
# coding: utf-8
# Note tf.multiply is equivalent to "*"
import tensorflow as tf
import numpy as np

# <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>
a = tf.placeholder("float") # Create a symbolic variable 'a'
# <tf.Tensor 'Placeholder_1:0' shape=<unknown> dtype=float	32>
b = tf.placeholder("float") # Create a symbolic variable 'b'

# <tf.Tensor 'Mul:0' shape=<unknown> dtype=float32>
y = tf.multiply(a, b) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(a*b, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
