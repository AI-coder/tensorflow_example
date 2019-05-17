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
    
    
 #tensorflow中几种获得模型中变量的信息
（1）tf.contrib.framework.get_variables_to_restore()#返回所有的variable
（2）tf.all_variables()#返回模型中所有的的变量
（3）tf.trainanle_variables()#返回模型中可以训练的变量


多分类采用与训练模型输出不匹配解决方法解决：
利用tf.contrib.framework.get_variables_to_restore()函数
variables_to_retore = tf.contrib.framework.get_variables_to_restore(exclude= ['wang'])
init = tf.global_variables_initializer()
savar = tf.train.savere(variables_to_restore)
with tf.Session()as sess:
    sess.run(init)
    saver.restore(sess,check_point)
#其中tf.contrin.framework.get_variables_to_restore()函数中
参数：exclude = ['wang']表示排除作用域‘wang’下的变量
include = [scope_name]表示恢复作用域scope_name下的变量
