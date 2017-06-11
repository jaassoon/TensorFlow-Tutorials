#!/usr/bin/env python

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101) #在指定的间隔内返回均匀间隔的数字。
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
print("trX: ",trX)
print("trY: ",trY)
X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")
# 线性回归：找出一个最优的线性对应关系使得给出输入得到的输出最接近真实值。

def model(X, w):
    return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)  # y= w * x

cost = tf.square(Y - y_model) # use square error for cost function square平方 FIXME：why use square?平方损失函数 又叫方差。
# Y与y_model的差值应该在小数级别，再算平方，使得乘积更小，以便用来做损失函数

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))  # It should be something around 2
