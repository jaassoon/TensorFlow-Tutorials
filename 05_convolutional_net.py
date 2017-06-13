#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 每次训练抓取的数据量
batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) # 按正态分布的随机值


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # relu函数其实就是一个max(0,x)，计算代价小很多
    # [ReLu(Rectified Linear Units修正线性单元)激活函数参考这里](http://www.mamicode.com/info-detail-873243.html)
    # [CNN训练Cifar-10技巧](http://www.cnblogs.com/neopenx/p/4480701.html)
    # [Cifar-10数据集](http://www.cs.toronto.edu/~kriz/cifar.html)
    # [改进的CNN网络，用于普适物体识别](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)
    # strides表示步长
    # tf.nn.conv2d 二维卷积网络 x,kernel,strides=(1, 1),padding='valid',
    # padding='SAME'表示通过填充0，使得输入和输出的形状一致。[参考这里](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    
    # Pooling(池化)层主要的作用是下采样，通过去掉Feature Map中不重要的样本，进一步减少参数数量。Pooling的方法很多，
    # 最常用的是Max Pooling(最大池化技术)。Max Pooling实际上就是在n*n的样本中取最大值，作为采样后的样本值
    # max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')

    # 为了减少过拟合程度，在输出层之前应用dropout技术（即丢弃某些神经元的输出结果）。我们创建一个placeholder来表示一个神经元的
    # 输出在dropout时不被丢弃的概率。Dropout能够在训练过程中使用，而在测试过程中不使用。TensorFlow中的tf.nn.dropout操作能够
    # 利用mask技术处理各种规模的神经元输出。
    # dropout(x, level, noise_shape=None, seed=None):
    # Sets entries in `x` to zero at random, while scaling the entire tensor.
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，第2、3维对应图片的宽和高，最后一维对应颜色通道的数目。（？第1维为什么是-1？）
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# 卷积层将要计算出32个特征映射(feature map)，对每个3 * 3的patch。它的权值tensor的大小为[3, 3, 1, 32]. 前两维是patch的大小，
# 第三维时输入通道的数目，最后一维是输出通道的数目。我们对每个输出通道加上了偏置(bias)。
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs  计算出32个特征映射(feature map)
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs  计算出64个特征映射(feature map)
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# 交叉熵 softmax_cross_entropy_with_logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
