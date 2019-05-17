import tensorflow as tf
with tf.variable_scope('xu'):
    a = tf.get_variable('a',shape = [2,2],dtype=tf.int32,initializer=tf.ones_initializer())
with tf.variable_scope('guang'):
    b = tf.get_variable('b', shape=[2, 2], dtype=tf.int32, initializer=tf.ones_initializer())
with tf.variable_scope('wang'):
    c = tf.get_variable('c', shape=[2, 2], dtype=tf.int32, initializer=tf.zeros_initializer())
with tf.variable_scope('chen'):
    d = tf.get_variable('d', shape=[2, 2], dtype=tf.int32, initializer=tf.ones_initializer())

init = tf.global_variables_initializer()
variables = tf.contrib.framework.get_variables_to_restore()
variables_to_restore =[v for v in variables if v.name.split('/')[0]!='wang']
saver = tf.train.Saver(variables_to_restore)
with tf.Session()as sess:
    sess.run(init)
    ckpt = tf.train.latest_checkpoint(r'log2/')
    saver.restore(sess,ckpt)
    print(sess.run([a,b,c,d]))
    #saver.save(sess,'log2/model.ckpt')

