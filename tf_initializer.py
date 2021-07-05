import tensorflow as tf

w = tf.Variable(tf.random_normal(shape=(1, 4), stddev=0.35), name='weight')
b = tf.Variable(tf.zeros([4]), name='bias')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(w.eval(), b.eval())

# partial initializer
with tf.Session() as sess:
    tf.variables_initializer([w]).run()
    print(w.eval())
