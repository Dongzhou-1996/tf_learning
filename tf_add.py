import tensorflow as tf
with tf.name_scope("AddExample"):
    a = tf.Variable(1.0, name='a')
    b = tf.Variable(2.0, name='b')
    c = tf.add(a, b, name='add')
    print(c)


