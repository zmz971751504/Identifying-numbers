import tensorflow as tf
import numpy as np
from PIL import Image
#create the model
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
#define the path
model_file=tf.train.latest_checkpoint('mnist_model')
#create 
saver=tf.train.Saver(max_to_keep=1)
saver.restore(sess,model_file)
#open image
image_path='number_two.jpg'
img = Image.open(image_path).convert('L')
flatten_img = np.reshape(img, 784)
x = np.array([1 - flatten_img])
#y = sess.run(sess.train.next_batch(100))
# 因为x只传入了一张图片，取y[0]即可
# np.argmax()取得独热编码最大值的下标，即代表的数字
print(image_path) 
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))