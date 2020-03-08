# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:10:51 2020

@author: zqh30
"""
import numpy as np
import tensorflow as tf

n_inputs = 3
n_neurons = 5
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
Wx = tf.Variable(tf.random_normal(shape = [n_inputs, n_neurons], dtype = tf.float32))  #random_normal表示从正态分布随机抽取
Wy = tf.Variable(tf.random_normal(shape = [n_neurons, n_neurons], dtype = tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype = tf.float32)) #偏置项
Y0 = tf.tanh(tf.matmul(X0,Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0,Wy) + tf.matmul(X1,Wx) + b) #Yt = Wx*Xt + Wy*Yt-1 + b
#和DNN区别在于，有一个X1的输入，即有不同的时序X输入
init = tf.global_variables_initializer()
X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]]) # t=0，且有四个instances
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})

print(Y0_val,'\n')
print(Y1_val)

n_steps = 2 #t=0; t=1
n_layers = 3 #这个是用来设定每一个RNN单元，有相同神经元数目的hidden layers
n_outputs = 1
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#X_seqs = tf.unstack(tf.transpose(X, perm = [1,0,2]))
#tf.unstack表示将矩阵分解（拆分,三维应该降到了两维，像前面X0,X1一样）
#tf.transpose中perm理解的链接，这里的意思是把shape[0]和shape[1]进行互换（使时间维度是第一维度） https://blog.csdn.net/qq_43088815/article/details/90116804 
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
#dropout_cell = tf.contrib.rnn.DroupoutWrapper(basic_cell, input_keep_prob = 0.5) dropout防止过拟合，但这个不管是训练还是使用模型都会应用，正常是训练时才用（前面加if is_training:）
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu),
        output_size = n_outputs)
multilayer_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units) for num_units in range (3)]) # 一般认为这样是不对的，因为会造成有相同的权重
                                            #  state_is_tuple = True)
# 将cell的定义放在layers的叠加过程中，这样不会报错
# https://cloud.tencent.com/developer/article/1443660  OutputProjectionWrapper作用链接，每个输出之上添加一个完全连接层（无激活函数，所有连接层权重和偏差项共享）
# OutputProjectionWrapper可将最终output向量降维到一个值
#basic_cell = tf.keras.layers.SimpleRNNCell(units = n_neurons) #http://www.ifunvr.cn/167.html  参数详解
#output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs,  #静态展开，不方便一般都用动态展开
#                                                dtype = tf.float32)
#outputs = tf.transpose(tf.stack(output_seqs),perm = [1,0,2])
seq_length = tf.placeholder(tf.int32, [None])  #shape = [None]的部分都是按输入实际shape中在这个维度的值
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32)
                                 #   sequence_length = seq_length)
# https://blog.csdn.net/u010960155/article/details/81707498  https://www.cnblogs.com/lovychen/p/9294624.html关于outputs和states的理解
# outputs含义，中间2表示两个时间步长，4表示X_batch中的4个instance，5表示设定的5个神经元。
# states含义，指的是每个单元的最终状态（也就是第一次出现零向量之前的非零向量）
# dynamic_rnn是指动态展开，避免内存不足的错误，且无需堆叠、转置、分解等操作

init = tf.global_variables_initializer()
X_batch = np.array([
        #t = 0    #t = 1
        [[0,1,2], [9,8,7]],
        [[3,4,5], [0,0,0]],  # [0，0，0]向量是为了满足数据输入的要求填充的，为了训练时避免训练到这个向量，加入sequence_length参数
        [[6,7,8], [6,5,4]],
        [[9,0,1], [3,2,1]],
        ])
seq_length_batch = np.array([2, 1, 2, 2])
with tf.Session() as sess:
    init.run()
    outputs_val,states_val = sess.run([outputs, states], 
                                      feed_dict = {X: X_batch})
                                                  # seq_length: seq_length_batch})
print(outputs_val) #有一个全零向量
print(states_val) #表示每个单元的最终状态，第二行的状态应是t=0时刻

#数据重塑，其实就是按要求的数据格式，reshape(-1,X,Y)即可，CNN是四维，RNN是三维，DNN只需两维
#minist数据集，原先有28*28的特征维度，在RNN中，因为输入数据是三维，拆分成n_input = 28； n_steps = 28
#将RNN用于时间序列的预测，只需要设定n_inputs = n_outputs = 1即可
#https://blog.csdn.net/bestrivern/article/details/90723524?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task 这个RNN原理讲的OK
import tensorflow as tf
  
batch_size=10
depth=128
 
inputs=tf.Variable(tf.random_normal([batch_size,depth]))
previous_state0=(tf.random_normal([batch_size,100]),tf.random_normal([batch_size,100]))
previous_state1=(tf.random_normal([batch_size,200]),tf.random_normal([batch_size,200]))
previous_state2=(tf.random_normal([batch_size,300]),tf.random_normal([batch_size,300]))

num_units=[100,200,300]
print(inputs)
 
cells=[tf.nn.rnn_cell.BasicLSTMCell(num_unit) for num_unit in num_units]
#正确调用MultiRNNCell的姿势
mul_cells=tf.nn.rnn_cell.MultiRNNCell(cells)

outputs,states=mul_cells(inputs,(previous_state0,previous_state1,previous_state2))

print(outputs.shape) #(10, 300)
print(states[0]) #第一层LSTM
print(states[1]) #第二层LSTM
print(states[2]) ##第三层LSTM

print(states[0].h.shape) #第一层LSTM的h状态,(10, 100)
print(states[0].c.shape) #第一层LSTM的c状态,(10, 100)
print(states[1].h.shape) #第二层LSTM的h状态,(10, 200)

#当时间步长很多时，RNN基本单元会对较前面的输入遗忘掉。
#LSTM，c(t)可理解为长时记忆状态；h(t)可理解为短时记忆状态
#c(t-1)经过遗忘门丢弃一些记忆，后在输入门中选择一些记忆。————有一部分成输出c(t)不经任何转换直接输出；
#另一部分经过tanh激活函数通过输出门得到短时记忆h(t)，同时也是这一时刻的单元的输出结果y(t)
#输入向量x(t)和前一时刻的短时状态h(t-1)作为输入传给四个全连接层
#FC1. g(t)将x(t)与h(t-1)解析，输出h(t)和y(t)，并一部分存储在长时状态中。（tanh激活函数）
#后三个为门控制器，采用logistic作激活函数。输入为0门关闭，为1门打开
#FC2. forget gate————f(t)，决定哪些长期记忆需要被擦除；
#FC3. input gate————i(t)，处理哪部分g(t)应该被添加到长时状态中，被称为部分存储
#FC4 output gate————o(t)，控制输出h(t),y(t) 
#P326 LSTM四个层的函数表示

#GRU，单元较LSTM简化，但性能实现相同。1. 长时状态和短时状态合并为一个向量h(t);2. 同一个门控制遗忘门和输出门，即当有新的记忆存储时，必须对其对应位置事先擦除该处记忆；3. 取消输出门，单元全部状态就是该时刻的单元输出




