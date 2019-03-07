# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network
 
CKPT_DIR = 'ckpt'
 
class Train(object):
    
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = input_data.read_data_sets('../mnist_data',one_hot = True)
        
    def train(self):
        batch_size = 64
        train_step = 10000
        step = 0
        #每隔1000步保存一次模型
        save_interval = 1000
        
        #tf.train.Saver用于保存训练的结果
        #max to keep 用于设置最多保存多少个模型
        #如果保存的模型超过这个值，最旧的模型被删除
        saver = tf.train.Saver(max_to_keep = 10)
        
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.get_checkpoint_state(CKPT_DIR):
            saver.restore(self.sess,ckpt.model_checkpoint_path)
            #读取网络中的global_step的值，即当前已经训练的次数
            step = self.sess.run(self.net.global_step)
            print('continue from')
            print('  -> Minibatch update : ',step)
            
        while step < train_step:
            x,label = self.data.train.next_batch(batch_size)
            _,loss = self.sess.run([self.net.train,self.net.loss],
                                   feed_dict = {self.net.x: x,self.net.label:label})
            
            step = self.sess.run(self.net.global_step)
            if step % 1000 == 0:
                print('第%6d步，当前loss: %.3f'%(step,loss))
                
            #模型保存在ckpt文件夹下
            #模型文件名最后会增加global_step的值，比如2000的模型文件名为model-2000
            if step % save_interval == 0:
                saver.save(self.sess,CKPT_DIR + '/model',global_step = step)
    
    def calculate_accuracy(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels
        acc = self.sess.run(self.net.accuracy,feed_dict = {self.net.x:test_x,self.net.label:test_label})
        
        print("准确率: %.3f，共测试了%d张图片 " % (acc, len(test_label)))
            
                
if __name__ == '__main__':
    model = Train()
    model.train()
    model.calculate_accuracy()