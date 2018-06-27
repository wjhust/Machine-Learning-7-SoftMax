#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:02:44 2018

@author: wangjian
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
###创建默认的session，后边就不须指定###
sess=tf.InteractiveSession()


################一.定义算法公式

####隐含层的参数设置
in_units=784#输入节点
h1_units=300#隐含层的输出节点
###将偏置为0，权重初始化为截断正态，标准差为0.1，
###tf.truncated_normal，产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
###初始化
w2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

##定义dropout的比率keep_prob
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

##定义模型结构#####
hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)####实现激活函数ReLU的隐含层
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)#使用dropout，随即将一部分节点置为0
#keep_prob是保留数据不被置为0的比例，训练时小于1，制造随机性防止过拟合；预测时=1，利用全部特征
y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)


##############二.定义损失函数和优化器选择
y_=tf.placeholder(tf.float32,[None,10])
#损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#优化器选择Adagrad，学习速率为0.3
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)




###############三.训练
tf.global_variables_initializer().run()
#keep_prob设置为0.75，保留75%
#每个batch包含100条样本，跑3000次
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})


##############四.准确率预测

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
