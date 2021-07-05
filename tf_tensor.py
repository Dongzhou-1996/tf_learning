import tensorflow as tf
a = tf.constant([1, 1])
b = tf.constant([2, 3])
c = tf.add(a, b)
with tf.Session() as sess:
    print("a[0] = {}, a[1] = {}".format(a[0].eval(), a[1].eval()))
    print("c.name = {}".format(c.name))
    print("c.value = {}".format(c.eval()))
    print("c.shape = {}".format(c.shape))
    print("a.consumers = {}".format(a.consumers()))
    print("b.consumers = {}".format(b.consumers()))
    print("[c.op]: \n {}".format(c.op))
