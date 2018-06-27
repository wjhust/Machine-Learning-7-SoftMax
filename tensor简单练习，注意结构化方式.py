#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:44:29 2018

@author: wangjian
"""
import tensorflow as tf
import numpy as np

#创建数据
#create data
x_data=np.random.rand(100).astype(np.float32)#tf的数据type是float32，需要定义好
y_data=x_data*0.1+0.3



#搭建模型
####create tensorflow structure start###
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
#权重是参数变量，由于这里权重是1维，则格式为【1】，范围是-1到1
biases=tf.Variable(tf.zeros([1]))#初始值为1
y=Weights*x_data+biases
####create tensorflow structure end###

#计算误差
loss=tf.reduce_mean(tf.square(y-y_data))#

#传播误差
optimizer=tf.train.GradientDescentOptimizer(0.5)#使用梯度下降，学习速率0.5
train=optimizer.minimize(loss)#优化器目标是-减少训练误差



#激活，相当于指向器
init=tf.global_variables_initializer()#结构初始化器
sess=tf.Session()
sess.run(init)#这里非常重要，sess可以看成指针，指向init


#训练
for step in range(201):#训练201步
    sess.run(train)    #训练指向train
    if step % 20 == 0:
        print step,sess.run(Weights),sess.run(biases)









