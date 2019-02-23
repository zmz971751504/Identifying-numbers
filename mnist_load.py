with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/my_model.meta') 
    new_saver.restore(sess, './model/my_model')