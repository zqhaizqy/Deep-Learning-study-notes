# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:58:58 2020

@author: zqh30
"""


import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

x = tf.Variable(3, name = 'x')
y = tf.Variable(2, name = 'y')
f = x*x*y + y + 2

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)


w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())  # 为了计算 y 和 z， w和x计算了两次

with tf.Session() as sess:
    y_val, z_val = sess.run([y,z])   #仅计算一次w和x，相当于节省了计算效率
    print(y_val)
    print(z_val)
#201页
import numpy as np
#from sklearn.datasets import iris
#housing = fetch_califonia_housing()

from sklearn import datasets
#清空sklearn环境下所有数据
datasets.clear_data_home()

#不加return命令，可以得到dictionary，并能知道数据每列含义
boston = datasets.load_boston()
boston.keys()
#载入波士顿房价数据
#X,y = datasets.load_boston(return_X_y=True)

m, n = boston.data.shape
#np.c_指将两矩阵按列方向合并，要求行数相同，这里是加入了全为1的偏置项
boston_data_plus_bias = np.c_[np.ones((m, 1)), boston.data]

X = tf.constant(boston_data_plus_bias, dtype = tf.float32, name = 'X')
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = 'y')
XT = tf.transpose(X)
#最小二乘法求解线性回归参数公式 θ = （XT·X）^-1·XT·y
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)

#梯度下降实现
from sklearn import preprocessing

m, n = boston.data.shape
#使用梯度下降时，需要将变量标准化/归一化
scaled_boston_data_plus_bias = preprocessing.scale(boston_data_plus_bias)
n_epochs = 1000 #迭代次数
learning_rate = 0.01

X = tf.constant(scaled_boston_data_plus_bias, dtype = tf.float32, name = 'X')
y = tf.constant(boston.target.reshape(-1, 1), dtype = tf.float32, name = 'y')
#初始化参数θ，线性回归中可以都是0，这里用随机向量
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = 'theta')#random_uniform[n+1,1]指产生n+1行1列的向量（行列不一定对。。。），后面的数字应该是每个值的范围是（-1，1）
y_pred = tf.matmul(X, theta, name = 'predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = 'mse') #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值,axis = 0:按行计算平均值；axis = 1:按列计算mean；不指定axis则是计算所有元素的mean
#这个梯度公式是仿照θ1的梯度下降推导公式得到的，由于初始θ0全为1，可以直接这样相乘
#gradients = 1/m * tf.matmul(tf.transpose(X), error) #梯度推导的话分子应该是1？不过问题不大，只相当于学习率的快慢而已
gradients = tf.gradients(mse, theta)[0] #tensorflow自动求解梯度的函数，使用了反向传播算法。它只需要通过N_output + 1次图遍历即可求解
#tf.gradients()输出格式为[[array[xxx],dytpe = float32],……]，因此要加[0]把第一个array取出，网上有代码不需要取，暂时没发现为什么。。
# tf.gradients()参数详解： https://www.cnblogs.com/luckyscarlett/p/10570938.html

#tf.assign —— tensorflow的独特迭代函数
'''
training_op = tf.assign(theta, theta - learning_rate * gradients) #tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)函数完成了将value赋值给ref的作用。其中：ref 必须是tf.Variable创建的tensor。链接：https://www.jianshu.com/p/fc5f0f971b14
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "; MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)  #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,	momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "; MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

#A = tf.placeholder(tf.float32, shape = (None, 3)) #创建占位符节点，仅用于输出运行时输出的数据
#B = A + 5
#这里特殊的是对A，B变量没必要初始化了
#with tf.Session() as sess:
#    B_val_1 = B.eval(feed_dict = {A: [[1,2,3]]}) #eval函数和run相似，feed_dict一般和place_holder联用，表示传入占位符的数组
#    B_val_2 = B.eval(feed_dict = {A: [[4, 5, 6], [7, 8, 9]]})
#print(B_val_1)
#print(B_val_2)


#将图形定义和训练统计信息写入TensorBoard将读取的日志目录(1. 给的网址改成localhost:6006;2. logdir中路径写全！！)
from datetime import datetime

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = r"./log"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01
#实现小批量梯度下降（随机森林之类的算法会用到）
X = tf.placeholder(tf.float32, shape = (None, n + 1), name = 'X')
y = tf.placeholder(tf.float32, shape = (None, 1), name = 'y')

batch_size = 50
n_batches = int(np.ceil(m / batch_size)) #np.ceil可以取到数组中大于等于该值的最小整数

#with tf.name_scope("loss") as scope: 名称作用域
#    error = y_pred - y
#    mse = tf.reduce_mean(tf.square(error), name="mse")


#有两个参数虽然函数中没有利用，但实际上是第几次迭代的指令传入
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) #这段作用应该就是呼应后面的step
    #print("know: ", know)
    indices = np.random.randint(m, size = batch_size) #随机生成batch_size个[0,m]范围的整数
    X_batch = scaled_boston_data_plus_bias[indices] #随机定下的第indices（一个数组）行数据
    y_batch = boston.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE',mse) #该节点求出MSE值并写入TensorBoard兼容的二进制日志
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) #将摘要写入日志目录的文件中

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 5 == 0:
                summary_str = mse_summary.eval(feed_dict = {X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
    best_theta = theta.eval()

file_writer.close()
print(best_theta)
