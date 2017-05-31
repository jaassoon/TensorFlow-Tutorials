import numpy as np
# A tensor's rank is its number of dimensions. 
# 3 # a rank 0 tensor; this is a scalar with shape [] #just value, no rank, rank=0
# [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
# rank=1
# 它只有在第一個矩陣的列數（column）和第二個矩陣的行數（row）相同時才有定義。
# The matmul operation only works on matrices (2D tensors).
vec = np.array([
			  20, 50
			 ])
a = np.array([
			  [20, 5],
			  [15, 10]
			 ])
w = np.array([[2.0, 1.0], 
			  [1.0, 4.0]
			 ])

# print(np.dot(a, w))
# [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
# print(np.sum(np.expand_dims(a, -1) * w , axis=0))
# equivalent result [2.6, 3]

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# print(node1, node2)

a = tf.constant(a, dtype=tf.float64)
vec = tf.constant(vec, dtype=tf.float64)
w = tf.constant(w)

with tf.Session() as sess:
  # they all produce the same result as numpy above
  # print(tf.constant(vec).eval()) 		# TypeError: List of Tensors when single Tensor expected
  # reference:http://blog.csdn.net/lenbow/article/details/52152766

  # print(sess.run(tf.rank(vec))) # 1
  # print(sess.run(tf.rank([1. ,2., 3.]))) # 1
  
  # print(tf.expand_dims(vec,0).eval()) # [[ 20.  50.]]
  # print(tf.expand_dims(vec,0)) 		# Tensor("ExpandDims:0", shape=(1, 2), dtype=float64)
  # print(sess.run(tf.rank(tf.expand_dims(vec,0)))) # 2

  # print(tf.expand_dims(vec,-1).eval()) # [[ 20.][ 50.]]
  # print(tf.expand_dims(vec,-1))		# Tensor("ExpandDims:0", shape=(2, 1), dtype=float64)
  # print(sess.run(tf.rank(tf.expand_dims(vec,-1)))) # 2

  # print(tf.expand_dims(vec,1).eval()) # [[ 20.][ 50.]]
  # print(tf.expand_dims(vec,1)) 		# Tensor("ExpandDims:0", shape=(2, 1), dtype=float64)
  # print(sess.run(tf.rank(tf.expand_dims(vec,1)))) # 2
  # ----------------------------------------------------------	 

  # print(w)		# Tensor("Const_4:0", shape=(2, 2), dtype=float64)
  # print(a)		# Tensor("Const_2:0", shape=(2, 2), dtype=float64)
  # print(tf.matmul(a, w).eval())	 # [[ 45.  40.][ 40.  55.]]

  # print (tf.expand_dims(a,0).eval()) # [[[ 20.   5.][ 15.  10.]]]
  # print (tf.expand_dims(a,0)) # Tensor("ExpandDims:0", shape=(1, 2, 2), dtype=float64)
  # print(tf.matmul(tf.expand_dims(a,0), w).eval())	# ValueError: Shape must be rank 2 but is rank 3 for 'MatMul' (op: 'MatMul') with input shapes: [1,2,2], [2,2].
  # print(tf.matmul(tf.expand_dims(vec,-1), w).eval()) #ValueError: Dimensions must be equal, but are 1 and 2 for 'MatMul' (op: 'MatMul') with input shapes: [2,1], [2,2].
  # print(tf.matmul(w,tf.expand_dims(vec,-1)).eval()) #[[  90.][ 220.]]
  # print(tf.matmul(tf.expand_dims(vec,0), w).eval()) # [[  90.  220.]]

  # print (tf.expand_dims(a,-1)) # Tensor("ExpandDims:0", shape=(2, 2, 1), dtype=float64)
  # print(sess.run(tf.rank(tf.expand_dims(a,-1)))) # 3
  # print (tf.expand_dims(a,-1).eval()) # [[[ 20.][  5.]][[ 15.][ 10.]]]
  # print(tf.matmul(tf.expand_dims(a,-1), w).eval()) # ValueError: Shape must be rank 2 but is rank 3 for 'MatMul' (op: 'MatMul') with input shapes: [2,2,1], [2,2].

  # print (tf.expand_dims(a,1).eval()) # [[[ 20.   5.]] [[ 15.  10.]]]
  # print(sess.run(tf.rank(tf.expand_dims(a,1)))) # 3
  # print(tf.expand_dims(a,1)) # Tensor("ExpandDims:0", shape=(2, 1, 2), dtype=float64)
  # ----------------------------------------------------------	 
  
  # print (tf.multiply(tf.expand_dims(a,-1), w).eval()) # [[[ 40.  20.]  [  5.  20.]][[ 30.  15.][ 10.  40.]]]
  # print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0)).eval()) # [[ 70.  35.][ 15.  60.]]
  # print((tf.reduce_sum(tf.multiply(a, tf.transpose(w)), axis=1)).eval())

  # Note tf.multiply is equivalent to "*"
  # print((tf.reduce_sum(tf.expand_dims(a,-1) * w, axis=0)).eval())
  # print((tf.reduce_sum(a * tf.transpose(w), axis=1)).eval())






