# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import tensorflow as tf
import math

import input_data
import resnet_utils
import resnet_v2

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

#%%
N_CLASSES = 5
BATCH_SIZE =64
IS_PRETRAIN = True
#%%   Test the accuracy on test dataset. 

def evaluate():
    with tf.Graph().as_default():
        

        log_dir = '/home/daijiaming/Galaxy/GalaxyNet/logs/train/'
        test_dir='/home/daijiaming/Galaxy/data3/testset/'
        test_label_dir='/home/daijiaming/Galaxy/data3/test_label.csv'
        n_test =2879
                
        val_image_batch, val_label_batch = input_data.read_galaxy11_test(data_dir=test_dir,
                                                                     label_dir=test_label_dir,
                                                                     batch_size= BATCH_SIZE)
        
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 3])
        y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES])
        
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
           logits, end_points,output0,output1 = resnet_v2.resnet_v2_26_2(x, N_CLASSES, is_training=True)
           
        correct = resnet_v2.num_correct_prediction(logits, y_)
        accuracy = resnet_v2.accuracy(logits, y_)
#        top_k_op = tf.nn.in_top_k(predictions=logits,targets=y_, k=1)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                num_step = int(math.ceil(n_test / BATCH_SIZE))
                num_sample = num_step*BATCH_SIZE
                step = 0
                total_correct = 0
                totall_acc=0.0
#                true_count = 0
                while step < num_step and not coord.should_stop():
                    test_images,test_labels=sess.run([val_image_batch, val_label_batch])
                    batch_correct,tes_acc,batch_logits = sess.run([correct,accuracy,logits],feed_dict={x:test_images,y_:test_labels}) 
#                    print('tes_acc = %.3f' % tes_acc)
                    total_correct += np.sum(batch_correct)
                    totall_acc= totall_acc+tes_acc
#                    print('totall_acc = %.3f' % totall_acc)
#                    true_count += np.sum(predictions)
                    if step==0:
                        a=test_labels 
                        b=batch_logits
                    if step >= 1:
                        a=np.concatenate((a,test_labels))
                        b=np.concatenate((b,batch_logits))
                    step += 1
#                precision = true_count / num_sample
                aver_acc=totall_acc/ num_step
                print('Aver acc = %.4f' % aver_acc)
                print('Total testing samples: %d' %num_sample)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.4f%%' %(100*total_correct/num_sample))
                np.savetxt('/home/daijiaming/Galaxy/GalaxyNet/labels2879.csv', a, delimiter = ',')
                np.savetxt('/home/daijiaming/Galaxy/GalaxyNet/logits2879.csv', b, delimiter = ',')
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
                
#%%


evaluate()





