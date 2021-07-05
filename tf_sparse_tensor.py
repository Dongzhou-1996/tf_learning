import tensorflow as tf
sp = tf.SparseTensor(indices=[[0, 2], [1, 3], [2, 1]],
		     values = [1, 3, 2],
		     dense_shape = [3, 4])
reduce_x = [tf.sparse_reduce_sum(sp),
	    tf.sparse_reduce_sum(sp, axis=1),
	    tf.sparse_reduce_sum(sp, axis=1, keep_dims=True),
	    tf.sparse_reduce_sum(sp, axis=[0, 1])]
with tf.Session() as sess:
    print(sp.eval())
    dense = tf.sparse_tensor_to_dense(sp)
    print(dense.eval())
    print(sess.run(reduce_x))
