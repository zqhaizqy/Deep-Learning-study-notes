# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:39:06 2020

@author: zqh30
"""
  #暂时不知道为什么只能运行一次
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

n_inputs = 28 * 28
n_hidden1 = 300 #第一个隐藏层有300个神经元
n_hidden2 = 100 
n_outputs = 10 #输出层有10类

mnist = input_data.read_data_sets("MNIST_data")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')

#tensorflow的DNN API直接调用

#feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
#dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = [300, 100], n_classes = 10,
#                                         feature_columns = feature_columns)
#dnn_clf.fit(x = X_train, y = y_train, batch_size = 50, steps = 1000)
#dnn_clf.evaluate(X_test, y_test)

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')

#加入dropout正则化技术
training = tf.placeholder_with_default(False, shape = (), name = 'training')
dropout_rate = 0.4 #每个神经元有0.4的概率被丢弃
X_drop = tf.layers.dropout(X, dropout_rate, training = training) #dropout 只能用于训练中

#最大范数正则化，作用：减少过拟合，减轻梯度消失问题（对每个神经元，它的输入连接权重w的l2范数小于等于r【最大范数超参数】）
def max_norm_regularizer(threshold, axes = 1, name = 'max_norm', collection = 'max_norm'):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm = threshold, axes = axes) #源码 https://blog.csdn.net/jinxin521125/article/details/77159112 ，不过只要知道是梯度裁剪的函数即可
        clip_weights = tf.assign(weights, clipped, name = name) #将clipped的值分配给weights，也就是更新了weights
        tf.add_to_collection(collection, clip_weights)  #将元素element添加到列表list_name中。
        return None  # 没有正则化损失项（真的要加我也不知道怎么加哈哈哈）
    return max_norm

scale = 0.001
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.elu,  #layers.dense就是全连接层的意思
                              	kernel_regularizer = max_norm_regularizer(threshold = 0.99),  #第一层使用前面定义的最大范数正则化器
                                name = 'hidden1')
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training = training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation = tf.nn.sigmoid,
                              kernel_regularizer = tf.contrib.layers.l1_regularizer(scale), #也可以用partial函数来设定新my_dense_layer函数
                              name = 'hidden2')
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training = training)

    logits = tf.layers.dense(hidden2_drop, n_outputs,
                                 name = 'outputs')
#tf.layers.dense()的类似函数
# def neuron_layer(X, n_neurons, name, activation = None):
#    with tf.name_scope(name):   #创建图层名称
#        n_inputs = int(X.get_shape()[1]) 
#        stddev = 2 / np.sqrt(n_inputs)
#        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev) #截断正态分布，随机初始化权重向量，并防止有大的权重使训练减慢
#        w = tf.Variable(init, name = 'weights')
#        b = tf.Variable(tf.zeros([n_neurons], name = 'biases')) #偏置项
#        z = tf.matmul(X, w) + b
#        if activation == 'relu':
#            return tf.nn.relu(z)
#        else:
#            return z



with tf.name_scope('loss'):  #softmax回归的交叉熵
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, #更高效，照顾了边界
                                                              logits = logits)
    #logits有无经过softmax函数转换无所谓，都可以计算loss，但经过转换后数据归一化了（每行向量和为1）
    base_loss = tf.reduce_mean(xentropy, name = 'avg_xentropy') #reduce_mean所有值求平均
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #有正则化项时，必须在损失计算时考虑进去
    loss = tf.add_n([base_loss] + reg_losses, name = 'loss') #tf.add_n表示将所有列表元素相加，reg_losses应该是加入到每一个列表的元素中去

learning_rate = 0.005

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1) #看是否与真值一致，in_top_k 用法链接https://www.cnblogs.com/logo-88/p/9099383.html
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0，1

init = tf.global_variables_initializer()

n_epoches = 20
batch_size = 50

clip_all_weights = tf.get_collection('max_norm') #返回名称为list_name的列表

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {training: True, X: X_batch,     #需要开启训练时，这里设定True即可
                                               y: y_batch})
            sess.run(clip_all_weights) #梯度裁剪必须每次在训练后运行这个东西
        acc_train = accuracy.eval(feed_dict = {X: X_batch,
                                               y: y_batch})
        acc_test = accuracy.eval(feed_dict = {X: mnist.test.images,
                                              y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)




