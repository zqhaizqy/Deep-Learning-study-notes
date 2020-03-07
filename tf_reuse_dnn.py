# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:01:36 2020

@author: zqh30
"""
#这个文件暂时还不能运行，需要其他数据集先训练前面的网络，再重新调用
#现在可运行，拿mnist数据集训练前面的网络，可用于调用分类其他数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

n_inputs = 28 * 28
n_hidden1 = 300 #第一个隐藏层有300个神经元
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 20
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

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu, 
                              name = 'hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation = tf.nn.elu,
                              name = 'hidden2')
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation = tf.nn.elu,
                              name = 'hidden3')
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation = tf.nn.elu,
                              name = 'hidden4')
    logits = tf.layers.dense(hidden2, n_outputs,
                                 name = 'outputs')

with tf.name_scope('loss'):  #softmax回归的交叉熵
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, #更高效，照顾了边界
                                                              logits = logits)
    loss = tf.reduce_mean(xentropy, name = 'loss') #所有值求平均

learning_rate = 0.005
#threshold = 0.99

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1) #看是否与真值一致，in_top_k 用法链接https://www.cnblogs.com/logo-88/p/9099383.html
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0，1

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #其他更快的优化方法：
    #1. 动量优化： 引入动量矢量m,超参数——动量β（通常设0.9），速度为学习率α/（1-β）
    #实现：optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
    #2. Nesrerov加速梯度: 与普通的动量优化区别是梯度在 ζ+βm处测量而非ζ处测量，减少振荡，更快收敛
    #实现： optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, 
                       #         momentum = 0.9, use_nesterov = True)
    #3. AdaGrad: 基本只能解决简单的二次问题或线性回归。优点是更新的结果更直接地指向全局最优，并且不需要花大量时间在调整学习率上
    #实现： 有AdagradOptimizer
    #4. RMSProp： 通过使用指数衰减，修正了AdaGrad提前终止的问题，在复杂问题时，效果较好
    #实现： optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, 
                       #         momentum = 0.9, decay = 0.9, epsilon = 1e-10) #decay是指数衰减的系数β，epsilon是避免被零除的平滑项
    #5. Adam优化：代表自适应矩估计，结合了动量优化和RMSProp的思想：就像动量优化一样，它追踪过去梯度的指数衰减平均值，就像RMSProp一样，它跟踪过去平方梯度的指数衰减平均值
    #实现： optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) 其他超参数一般都用默认值
    #PS：二阶偏导数的优化算法，理论上精准，实际上无法用于深度神经网络，因为慢，而且不适合内存
    training_op = optimizer.minimize(loss)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope = 'hidden[12]') #用正则表达式，仅抽取第一层和第二层来reuse（DNN法的迁移学习）
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 10
batch_size = 50

with tf.Session() as sess:
    init.run()
    #restore_saver.restore(sess, "./my_model_final.ckpt") #前面的网络应该需要先保存，路径为"./my_model_final.ckpt"
    
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X: X_batch,
                                               y: y_batch})
        acc_train = accuracy.eval(feed_dict = {X: X_batch,
                                               y: y_batch})
        acc_test = accuracy.eval(feed_dict = {X: mnist.test.images,
                                              y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = restore_saver.save(sess, "./my_model_final.ckpt")


#冻结较低层，指冻结较低层的权重，也达到fine-tune迁移学习的目的。注意，也是导入了一个已经训练好的模型
#方式：在train scope里面，加：
#train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
#    scope = 'hidden[34]|outputs') 竖线表示或。只拿了四层做举例
#training_op = optimizer.miniize(loss, var_list = train_vars)

#	https://github.com/tensorflow/models  tensorflow的各种神经网络模型链接