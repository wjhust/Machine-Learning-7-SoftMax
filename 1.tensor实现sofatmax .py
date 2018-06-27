#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#2.导入数据集mnist
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


#2.查看数据集：
print mnist.train.images.shape,mnist.train.labels.shape
print mnist.test.images.shape,mnist.test.labels.shape
print mnist.validation.images.shape,mnist.validation.labels.shape



#3.数据预处理
sess=tf.InteractiveSession()#用来激活的
x=tf.placeholder(tf.float32,[None,784])  #placeholder是用来设定输入的格式
#placeholder第一个参数是数据类型，第二个参数是shape，即数据尺寸
##发现使用tf时都要先定义好格式和维度
w=tf.Variable(tf.zeros([784,10]))#创建权重的对象
b=tf.Variable(tf.zeros([10]))#创建偏差的对象，10维
y=tf.nn.softmax(tf.matmul(x,w)+b)#softmax函数



#4.loss函数
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#损失函数，多分类问题，使用交叉熵作为loss，使用SGD进行训练，步长0.5
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#设置训练步骤，学习速率为0.5，最小化loss函数
tf.global_variables_initializer().run()#全局初始化
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print '@',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})

