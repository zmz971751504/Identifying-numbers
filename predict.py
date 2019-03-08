# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image
from model import Network
 
CKPT_DIR = 'ckpt'
 
 
class Predict(object):
    
    def __init__(self):
        #清除默认图的堆栈，并设置全局图为默认图
        #若不进行清楚则在第二次加载的时候报错，因为相当于重新加载了两次
        tf.reset_default_graph() 
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        #加载模型到sess中
        self.restore()
        print('load susess')
    
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('未保存模型')
        
    def predict(self,image_path):
        #读取图片并灰度化
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img,784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y,feed_dict = {self.net.x:x})
        
        print(image_path)
        print(' Predict digit',np.argmax(y[0]))
        
        
if __name__ == '__main__':
    model = Predict()
    model.predict('mnist_picture/0.jpg')
    model.predict('mnist_picture/1.jpg')
    model.predict('mnist_picture/2.jpg')
    model.predict('mnist_picture/3.jpg')
    model.predict('mnist_picture/4.jpg')
    model.predict('mnist_picture/5.jpg')
    model.predict('mnist_picture/6.jpg')
    model.predict('mnist_picture/7.jpg')
    model.predict('mnist_picture/8.jpg')
    model.predict('mnist_picture/9.jpg')