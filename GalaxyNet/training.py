# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import tensorflow as tf
import input_data
import resnet_utils
import resnet_v2
slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope
#%%
N_CLASSES = 5
BATCH_SIZE =128
learning_rate = 0.1
MAX_STEP = 100000   # it took me about one hour to complete the training.

#%%   Training
def train():
    
    train_dir='/home/daijiaming/Galaxy/data3/trainset/'
    train_label_dir='/home/daijiaming/Galaxy/data3/train_label.csv'
    test_dir='/home/daijiaming/Galaxy/data3/testset/'
    test_label_dir='/home/daijiaming/Galaxy/data3/test_label.csv'
    
    train_log_dir = '/home/daijiaming/Galaxy/GalaxyNet/logs/train/'
    val_log_dir = '/home/daijiaming/Galaxy/GalaxyNet/logs//val/'
    
    tra_image_batch, tra_label_batch = input_data.read_galaxy11(data_dir=train_dir,
                                                                label_dir=train_label_dir,
                                                                batch_size= BATCH_SIZE)
    val_image_batch, val_label_batch = input_data.read_galaxy11_test(data_dir=test_dir,
                                                                     label_dir=test_label_dir,
                                                                     batch_size= BATCH_SIZE)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES])
#    keep_prob=tf.placeholder(tf.float32)
                            
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      logits, end_points,output0,output1 = resnet_v2.resnet_v2_26_2(x, N_CLASSES, is_training=True)
       
    loss = resnet_v2.loss(logits, y_)
#    rmse=resnet_v2.compute_rmse(logits, y_)
    accuracy = resnet_v2.accuracy(logits, y_)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = resnet_v2.optimize(loss, learning_rate, my_global_step)
    
   
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   
       
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss,tra_acc,summary_str = sess.run([train_op,loss, accuracy,summary_op],feed_dict={x:tra_images, y_:tra_labels})
            
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, tra_loss: %.4f, tra_accuracy: %.2f%%' % (step, tra_loss, tra_acc))
#                summary_str = sess.run(summary_op,feed_dict={x:tra_images, y_:tra_labels})
                tra_summary_writer.add_summary(summary_str, step)
                
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc, summary_str = sess.run([loss, accuracy,summary_op],feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, test_loss = %.4f, test_accuracy = %.2f%%  **' %(step, val_loss, val_acc))
#                summary_str = sess.run([summary_op],feed_dict={x:val_images,y_:val_labels})
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()



train()
