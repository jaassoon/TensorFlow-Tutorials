#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 每次训练抓取的数据量
batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) # 按正态分布的随机值

# cnnLayer 定义卷积网络
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # relu函数其实就是一个max(0,x)，计算代价小很多
    # [ReLu(Rectified Linear Units修正线性单元)激活函数参考这里](http://www.mamicode.com/info-detail-873243.html)
    # [CNN训练Cifar-10技巧](http://www.cnblogs.com/neopenx/p/4480701.html)
    # [Cifar-10数据集](http://www.cs.toronto.edu/~kriz/cifar.html)
    # [改进的CNN网络，用于普适物体识别](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)
    # strides表示步长
    # tf.nn.conv2d 二维卷积网络 x,kernel,strides=(1, 1),padding='valid', # x为输入；kernel为特征检测器filter
    # 忽略边缘(padding=’VALID’)：将多出来的边缘直接省去。
    # 保留边缘(padding=’SAME’)：将特征图的变长用0填充为2的倍数，然后再池化（一般使用这种方式）。
    # padding='SAME'表示通过填充0，使得输入和输出的形状一致。[参考这里](http://www.jeyzhang.com/tensorflow-learning-notes-2.html)
    # 输入28*28矩阵，经过3*3的滤波器，得到14*14矩阵? # （28-3+2*1)/2+1=14 W2=(W1-F+2P)/S+1
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    
    # Pooling(池化)层主要的作用是下采样，通过去掉Feature Map中不重要的样本，进一步减少参数数量。Pooling的方法很多，
    # 最常用的是Max Pooling(最大池化技术)。Max Pooling实际上就是在n*n的样本中取最大值，作为采样后的样本值
    # 池化函数可以逐渐降低输入表示的空间尺度。特别地，池化:
    # 使输入表示(特征维度)变得更小，并且⺴络中的参数和计算的数量更加可控的减小，因此，可以控制过拟合 使⺴络对于输入图像中更小的变化、
    # 冗余和变换变得不变性(输入的微小冗余将不会改变池化的输出——因为我们在局部邻域中使用了最大化/平均值的操作。 帮助我们获取图像最大程度
    # 上的尺度不变性(准确的词是“不变性”)。它非常的强大，因为我们可以检测图像中的物体，无论它们位置在哪里
    # max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    # TODO ksize和strides如何确定？
    # 步长设置为2，使得矩阵变为14*14
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')

    # 为了减少过拟合程度，在输出层之前应用dropout技术（即丢弃某些神经元的输出结果）。我们创建一个placeholder来表示一个神经元的
    # 输出在dropout时不被丢弃的概率。Dropout能够在训练过程中使用，而在测试过程中不使用。TensorFlow中的tf.nn.dropout操作能够
    # 利用mask技术处理各种规模的神经元输出。
    # Dropout的好处，每次丢掉随机的数据，让神经网络每次都学习到更多，但也需要知道，这种方式只在我们有的训练数据比较少时很有效
    # dropout(x, level, noise_shape=None, seed=None):
    # Sets entries in `x` to zero at random, while scaling the entire tensor.
    # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0. 
    # The scaling is so that the expected sum is unchanged.
    # 输出的非0元素是原来的 “1/keep_prob” 倍
    # p_keep_conv or p_keep_prob(probability)保留概率
    # keep_prob是保留概率，即我们要保留的RELU的结果所占比例，
    l1 = tf.nn.dropout(l1, p_keep_conv)

    # w2 = init_weights([3, 3, 32, 64]) # 从之前的32个特征值中，每个再产生64个特征值
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))

    # 输入14*14矩阵，经过3*3的滤波器，得到7*7矩阵 # （14-3+2*1)/2+1=7
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')

    # p_keep_conv: 0.8, p_keep_hidden: 0.5
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # 从之前的64个特征值中，每个再产生128个特征值
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))

    # 输入7*7矩阵，经过3*3的滤波器，得到4*4矩阵 # （7-3+2*1)/2+1=4
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    # l3 为4*4矩阵，将其reshape成2048的矩阵？ 128组4*4矩阵的全连接网络FC为128*4*4矩阵 
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    # w4 FC将产生625个输出：第一个隐藏层：由2048个输入产生625个输出
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    # p_keep_hidden 全连接网络隐藏层使用的保留概率。
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    # output层：对625个输入产生10个输出 
    pyx = tf.matmul(l4, w_o)
    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，第2、3维对应图片的宽和高，最后一维对应颜色通道的数目。（？第1维为什么是-1？）
# 通道为1，表示灰度图； 通道为3，表示红绿蓝三色图。
# pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # -1 表示每次抓取的数据量将会基于样本的数量动态计算。
# In the `reshape()` operation above, the `-1` signifies that the *`batch_size`*
# dimension will be dynamically calculated based on the number of examples in our
# input data. Each example has 7 (`pool2` width) * 7 (`pool2` height) * 64
# (`pool2` channels) features, so we want the `features` dimension to have a value
# of 7 * 7 * 64 (3136 in total). The output tensor, `pool2_flat`, has shape
# <code>[<em>batch_size</em>, 3136]</code>.
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# 卷积层将要计算出32个特征映射(feature map)，对每个3 * 3的patch。它的权值tensor的大小为[3, 3, 1, 32]. 前两维是patch的大小，
# 第三维是输入通道的数目，最后一维是输出通道的数目。我们对每个输出通道加上了偏置(bias)。
# 32 表示将通过正态分布随机产生32个滤波器，从而产生32个特征。
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs  计算出32个特征映射(feature map)

# 为了使得网络有足够深度，我们重复堆积一些相同类型的层。第二层将会有64个特征，对应每个3 * 3的patch。
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs  计算出64个特征映射(feature map)
w3 = init_weights([3, 3, 64, 128])    # 3x3x64 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
# p_keep_conv: 0.8, p_keep_hidden: 0.5
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# 交叉熵cross_entropy softmax_cross_entropy_with_logits
# [对交叉熵的论述博文](http://colah.github.io/posts/2015-09-Visual-Information/)
# 损失函数
# tf.reduce_mean 沿着指定维度求平均值。最终结果为压缩到该维度的矢量。不指定维度，则结果为单个值。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
# RMSPropOptimizer 优化算法
# Optimizer that implements the RMSProp algorithm
# See the [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
# RMSPropOptimizer(optimizer.Optimizer)
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，
# 因此最大值1所在的索引位置就是类别标签，argmax(x, axis=-1)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size), # batch_size=128 每次训练抓取的数据量
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
