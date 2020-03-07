# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:18:56 2020

@author: zqh30
"""

from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
dataset = np.array([china, flower], dtype = np.float32)
batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape = (7, 7, channels, 2), dtype = np.float32)  #两个7*7的卷积核
filters[:, 3, :, 0] = 1 #垂直过滤线
filters[3, :, :, 1] = 1 #水平过滤线

# 关于深度学习中四维数组的简单解释 https://blog.csdn.net/holmes_MX/article/details/82813865?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
#test_filter = np.zeros(shape = (7,7,2))
#test_filter[3, :, 1] = 1
#test_filter[:, 3, 0] = 1
X = tf.placeholder(tf.float32, shape = (None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides = [1, 2, 2, 1], padding = 'SAME')
#strides是步幅参数，中间两个是垂直和水平步幅，第一个（决定是否跳过一些实例）和最后一个（决定是否跳过上一层的特征映射）一般为1
#padding = 'SAME'时，必要时使用零填充  链接：https://blog.csdn.net/lujiandong1/article/details/53728053?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict = {X: dataset})

plt.imshow(output[1, :, :, 0], cmap = 'gray') #第二张图片的第二个特征
#两个数组算卷积的要求暂且不论，但最后output的结果batch_size, height, width, channels都有一定的压缩（不是所有的像素点都需要学习的意思），【channel是因为两个filter的缘故】
plt.show()


X_pool = tf.placeholder(tf.float32, shape = (None, height, width, channels))
max_pool = tf.nn.max_pool(X_pool, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') #还有avg_pool
#ksize参数规定：1. 第一个参数必须为1；2. 第二个参数和第三个参数均为1或第四个参数为1
#池化层作用，对输入图像进行抽样（收缩），减少计算负担，内存使用量和参数数量（限制过拟合）
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict = {X_pool: dataset})

plt.imshow(output[0].astype(np.uint8)) 
plt.show()

#关于网络架构构建，P299-306
#https://blog.csdn.net/clover_my/article/details/102826498 CNN网络架构图解

#import numpy as np
#a= np.array([0,1,2,3,4,5,6,7])
#c = a.reshape(2,-1)

