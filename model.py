# -*- coding: utf-8 -*-
import tensorflow as tf
 
#构建网络图
class Network(object):
    
    def __init__(self):
        
        self.learning_rate = 0.001
        #几率已经训练的次数
        self.global_step = tf.Variable(0,trainable = False)
        
        self.x = tf.placeholder(tf.float32,[None,784])
        self.label = tf.placeholder(tf.float32,[None,10])
        
        self.w = tf.Variable(tf.zeros([784,10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x,self.w) + self.b)
        
        self.loss = -tf.reduce_mean(self.label * tf.log(self.y) + 1e-10)
        
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,global_step = self.global_step)
        
        predict = tf.equal(tf.argmax(self.label,1),tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))