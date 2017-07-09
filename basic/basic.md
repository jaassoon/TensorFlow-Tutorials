#### 矩阵乘法的解释[参考这里](https://www.zhihu.com/question/21351965)
在左边的这个矩阵的每一行，都代表了一种价目表；在右边的矩阵的每一列，都代表了一种做饭方式。那么所有可能的组合所最终产生的花费，则在结果矩阵中表示出来了。
它只有在第一個矩陣的列數（column）和第二個矩陣的行數（row）相同時才有定義。

source ~/tensorflow/bin/activate

easy_install virtualenv

sudo easy_install pip

#Installed /Library/Python/2.7/site-packages/pip-9.0.1-py2.7.egg

sudo pip install --upgrade virtualenv

#Requirement already up-to-date: virtualenv in /Library/Python/2.7/site-packages/virtualenv-15.1.0-py2.7.egg


virtualenv --system-site-packages -p python3 /Users/jason/tensorflow

pycharm

canonical 权威的，被广泛认可的
MNIST the canonical dataset for trying out a new machine learning toolkit
手写数字识别 (MNIST),
单词嵌套 (word embedding)
循环神经网络 (Recurrent Neural Network, 简称 RNN)
序列到序列模型 (Sequence-to-Sequence Model)

Softmax Regressions　Softmax回归
 There are many predefined types like 
 linear regression,
 logistic regression, 
 linear classification, 
 logistic classification, and
 many neural network classifiers and regressors.

ML Beginners ML：Machine Learning
A tensors rank is its number of dimensions. 使用rank来表示数据的维度  [[1., 2., 3.], [4., 5., 6.]] 表示二维数据‘
The central unit of data in TensorFlow is the tensor.  使用 tensor 表示数据
A tensor consists of a set of primitive values shaped into an array of any number of dimensions
a tensor (an n-dimensional array)
TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. 自动调整参数，优化
linear regression 线性回归，取最优参数
probabilities　概率
bias 偏移量
用 tf.reduce_sum 计算张量的所有元素的总和。
使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的
TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）
是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用
其他的算法。
使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步
的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的
总体特性

首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，
因此最大值1所在的索引位置就是类别标签，
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 取平均值
where 
 is the weights 权重 and W
 is the bias 偏移量 for class  b
 y = tf.nn.softmax(tf.matmul(x, W) + b)  矩阵乘法 tf.matmul()
 One very common, very nice function to determine the loss of a model is called "cross-entropy."  交叉熵
, and 
 is an index for summing over the pixels in our input image 
. We then convert the evidence tallies into our predicted probabilities 
 using the "softmax" function:

# loss 损失函数
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer 优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss) #目标： 令损失最小
tf.contrib.learn is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:
running training loops
running evaluation loops
managing data sets
managing feeding



