#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    # 按正态分布产生随机值
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) 


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked 
    					   # in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)
# tf.reduce_mean 求平均值
# tf.nn.softmax_cross_entropy_with_logits 
# tf.nn.softmax_cross_entropy_with_logits 在模型的非归一化预测结果上做 Softmax，并对所有类求和。tf.reduce_mean 是对这些和求平均
# 计算batch维度（第一维度）下交叉熵（cross entropy）的平均值，将将该值作为总损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) 	# compute mean cross entropy 
																						# (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，
# 因此最大值1所在的索引位置就是类别标签，
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 取平均值
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
