import numpy as np
import tensorflow as tf
# shape(2,2)
a = np.array([
			  [20.0, 5.0],
			  [15.0, 10.0]
			 ])
# shape(2,2)
w = np.array([[2.0, 1.0], 
			  [1.0, 4.0]
			 ])

# TODO np.dot()
print(np.dot(a, w))
# [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
# print(np.sum(np.expand_dims(a, -1) * w , axis=0))
# equivalent result [2.6, 3]

with tf.Session() as sess:
  # reference:http://www.jianshu.com/p/00ab12bc357c
  # reference:[Tensorflow一些常用基本概念与函数（1）](http://blog.csdn.net/lenbow/article/details/52152766)
  # [from_tensor](https://www.tensorflow.org/api_docs/python/tf/transpose)
  # 调换tensor的维度顺序
  # 按照列表perm的维度排列调换tensor顺序，
  # default,perm为(n-1…0)
  # ‘x’ is [[1 2 3],[4 5 6]]
  # tf.transpose(x) ==> [[1 4], [2 5],[3 6]]
  # Equivalently
  # tf.transpose(x, perm=[1, 0]) ==> [[1 4],[2 5], [3 6]]
  # 解释：将a进行转置，并且根据perm参数重新排列输出维度。
  # 输出数据tensor的第i维将根据perm[i]指定。比如，如果perm没有给定，那么默认是perm = [n-1, n-2, ..., 0]，其中rank(a) = n。默认情况下，
  # 对于二维输入数据，其实就是常规的矩阵转置操作。 
  print(tf.transpose(a).eval()) # [[ 20.  15.][  5.  10.]]
  print(tf.transpose(a,perm=[1, 0]).eval()) # [[ 20.  15.][  5.  10.]]

  print(tf.transpose(w).eval()) # [[ 2.  1.][ 1.  4.]]
  print(tf.multiply(a, tf.transpose(w)).eval()) # [[ 40.   5.] [ 15.  40.]]
  print((tf.reduce_sum(tf.multiply(a, tf.transpose(w)), axis=1)).eval()) # [ 45.  55.]







