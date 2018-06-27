#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:03:45 2018

@author: wangjian
"""


#使用mnist数据集，两个卷基层和一个全连接层

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()

##初始化权重和偏置，加一些随机噪声
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    #截断正态分布，标准差为0.1
    return tf.Variable(initial)

def bias_variable(shape):#偏置标准差设为0.1避免死亡节点
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
#2维的卷积函数，[卷积核尺寸，卷积核尺寸，channel数，卷积核数量]，padding是边界的处理方式
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#最大池化函数，使用2x2池化，即将2X2的像素降为1X1的像素，最大池化函数会保留原始像素中灰度最高的那一个
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')





#####设计输入，x是特征，y是真是的label
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])#需要将1X784的形式转化为28X28的形式



####第一个卷积层，5x5的卷积核，1个颜色通道，32个核，提取32个特征
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#卷积操作再使用relu非线性化
h_pool1=max_pool_2x2(h_conv1)
#2X2池化后缩小为原来的1/4，图片🈶️28x28变成7x7

######第二个卷积层，5x5尺寸，32个颜色通道,64个核，即会提取64个特征,输出的tensor为7x7x64
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#减轻过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)



###定义损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


###定义准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



###开始训练

tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print "step %d,training accuracy%g"%(i,train_accuracy)
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})



##输出测试
print "test accuracy %g"%accuracy.eval(feed_dict={
        x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
    















