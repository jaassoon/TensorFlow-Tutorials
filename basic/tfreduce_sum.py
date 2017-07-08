import numpy as np
import tensorflow as tf
# reduction 简单粗暴的说就是，n 维的数据中，把某一个维度上这一序列的数缩减到一个（比如求它们的和或者平均值），这样就降低了一个维度，所以叫 reduce，
# 比如 map-reduce 也差不多是这个意思。目前这个 reduction_indices 参数改名叫 axis ，这样就直白多了，代表 reduce 
# 是沿着哪个轴的方向进行的。
# shape(2,)
vec = np.array([
			  20.0, 50.0
			 ])
# shape(2,2)
a = np.array([
			  [20.0, 5.0],
			  [15.0, 10.0]
			 ])
# shape(2,2)
w = np.array([[2.0, 1.0], 
			  [1.0, 4.0]
			 ])

with tf.Session() as sess:
  # https://www.tensorflow.org/api_docs/python/tf/expand_dims
  # 从左至右依次为0，1，2　…　负数则循环推算
  # <tf.Tensor 'ExpandDims_3:0' shape=(2, 2, 1) dtype=float64>
  tf.expand_dims(a,-1)
  # shape(2,2,1)*shape(2,2) 
  # <tf.Tensor 'Mul_4:0' shape=(2, 2, 2) dtype=float64>
  tf.multiply(tf.expand_dims(a,-1), w)
  # Evaluates the value of a variable. like run()?
  # array([[[ 40.,  20.],
  #       [  5.,  20.]],

  #      [[ 30.,  15.],
  #       [ 10.,  40.]]])
  tf.multiply(tf.expand_dims(a,-1), w).eval()
  # reference:https://www.zhihu.com/question/51325408
  # https://www.tensorflow.org/api_docs/python/tf/reduce_sum 
  (tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w))).eval()  # 180
  (tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0)).eval() # [[ 70.  35.][ 15.  60.]]
  (tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=[0,1])).eval() # 
  print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=1)).eval()) # [[45 40][40 55]]
  print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=2)).eval()) # [[60 25][45 50]]
  # keep_dims: If true, retains reduced dimensions with length 1.
  (tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0,keep_dims=True)).eval() # 






