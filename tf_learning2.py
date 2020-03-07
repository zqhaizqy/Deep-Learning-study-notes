# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:32:06 2020

@author: zqh30
"""
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

n_inputs = 28 * 28
n_hidden1 = 300 #第一个隐藏层有300个神经元
n_hidden2 = 100 
n_outputs = 10 #输出层有10类

mnist = input_data.read_data_sets("MNIST_data")

batch_norm_momentum = 0.9
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')
training = tf.placeholder_with_default(False, shape = (), name = 'training') # 给Batch_norm加palceholder

with tf.name_scope('dnn'):
    he_init = tf.contrib.layers.variance_scaling_initializer() # 权重初始化，且做标准化操作
    my_batch_norm_layer = partial( # partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数。
            tf.layers.batch_normalization,  # https://www.jianshu.com/p/437fb1a5823e 函数参数详解，这里也可以看到training默认值是False，
            #所以前面placeholder_with_default里的input是False，并且需要更新平均误差
            training = training,
            momentum = batch_norm_momentum) # 用在训练时，滑动平均的方式计算滑动平均值moving_mean和滑动方差moving_variance。
    #公式为: moving_average_value * momentum + value * (1 - momentum),其中value为当前batch的平均值或方差，moving_average_value为滑动均值或滑动方差。
    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer = he_init #保证每层是标准化的初始
            )
    hidden1 = my_dense_layer(X, n_hidden1, name = 'hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1)) #elu是一个和指数函数类似的激活函数
    hidden2 = my_dense_layer(bn1, n_hidden2, name = 'hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logists_before_bn = my_dense_layer(bn2, n_outputs, name = 'outputs')
    logists = my_batch_norm_layer(logists_before_bn)

with tf.name_scope('loss'):  #softmax回归的交叉熵
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, #更高效，照顾了边界
                                                              logits = logists)
    loss = tf.reduce_mean(xentropy, name = 'loss') #所有值求平均

#threshold = 0.99 #梯度裁剪程度，范围-1~1，是超参数
with tf.name_scope('train'):
    #实现学习率衰减，开始时下降快，然后减少学习率防止偏离和振荡
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10 #表示每r步下降10倍的学习率
    global_step = tf.Variable(0, trainable = False, name = 'global_step')
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                         decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
#grads_and_vars = optimizer.compute_gradients(loss)
#capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
#       for grad, var in grads_and_vars] #梯度裁剪核心步骤
    training_op = optimizer.minimize(loss,global_step = global_step)
#training_op = optimizer.minimize(capped_gvs)
#P249, 有无需在训练评估时加入更新操作的办法
    
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logists, y, 1) #看是否与真值一致，in_top_k 用法链接https://www.cnblogs.com/logo-88/p/9099383.html
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0，1

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #额外的更新操作    

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], 
                     feed_dict = {X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict = {X: X_batch,
                                               y: y_batch})
        acc_test = accuracy.eval(feed_dict = {X: mnist.test.images,
                                              y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    #save_path = saver.save(sess, "./my_model_final.ckpt")



