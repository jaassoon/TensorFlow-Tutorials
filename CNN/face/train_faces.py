import tensorflow as tf
# pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = './my_faces1'
my_faces_path = '/Users/jason/Documents/tensorTrain1112/data/train/cats'
other_faces_path = '/Users/jason/Documents/tensorTrain1112/data/train/dogs'
# other_faces_path = './other_faces1'
# size = 256
# size = 128
# size = 64
size = 32
fc1Size=int(8*(size/64))

imgs = []
labels = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labels.append(path)

readData(my_faces_path)
readData(other_faces_path)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
labels = np.array([[0,1] if label == my_faces_path else [1,0] for label in labels])
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labels, test_size=0.05, random_state=random.randint(0,100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取100张图片
batch_size = 50
# batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# keep_prob_5:0.5,keep_prob_75:0.75 保留概率
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) # 按正态分布的随机值

def biasVariable(shape):
    return tf.Variable(tf.random_normal(shape)) # 按正态分布的随机值

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层 输入通道3 红蓝绿； 输出通道32: 产生32个特征值
    # W1 定义滤波器 3*3矩阵
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化 
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层 8*8矩阵64组；
    Wf = weightVariable([fc1Size*fc1Size*64, 512])
    # Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, fc1Size*fc1Size*64])
    # drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层 512个输入，产生2个输出
    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out
out = cnnLayer()
# softmax:在分类结果出来之后，计算每个分类的概率。比如一批取32个样本，
# 每个样本的结果保存在[0,1]类似的变量中，每个数组下标对应一个分类，
# 这样在按数组下标来算概率的方法得到每个类别的概率分布
y_pred = tf.nn.softmax(out)
# arg Max：按行或列取该行或列最大值的下标，0表示按列，1表示按行；无论如何最终结果都是一行
# 这里按行取，从而得到每个样本概率最大的值所在的下标
# 形如[1 0 0 1 1 1 1 ...32个位置]
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y_)
# reduce_mean：按指定维度求平均值，不指定维度，则求所有值的平均值，为一个数字
cost = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)
# 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
session = tf.Session()
session.run(tf.global_variables_initializer())
def cnnTrain():
    # 数据保存器的初始化
    saver = tf.train.Saver()
    print('num_batch=',num_batch)
    # 自定义迭代次数i
    for n in range(1):
    # for n in range(20):
        for i in range(num_batch):
            batch_x = train_x[i*batch_size : (i+1)*batch_size]
            batch_y = train_y[i*batch_size : (i+1)*batch_size]
            # 开始训练数据，同时训练三个变量，返回三个数据
            _,loss = session.run([train_step, cost],
                                       feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
            if i % 50 == 0:
                # 获取测试数据的准确率
                acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0},session=session)
                print('i= ',i,'accuracy= ', acc)
                feed_dict_train =    {x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75}
                feed_dict_validate = {x: test_x,y_: test_y, keep_prob_5:0.5,keep_prob_75:0.75}
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                valid_acc = session.run(accuracy, feed_dict=feed_dict_validate)
                msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
                print(msg.format(i, acc, valid_acc, loss))
                saver.save(session, './model/train_faces.model', global_step=i)

cnnTrain()
saver = tf.train.Saver()

