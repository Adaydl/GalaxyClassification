# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import random

def get_images(data_dir,label_dir):
    
    img_width=random.randint(170,241)
    img_height=img_width
#    img_depth = 3
    
    
    
    with tf.name_scope('input'):
        
        label=pd.read_csv(label_dir)
        index1=label['GalaxyID']

        label_list=label['class']
        label_list=list(label_list)
        image_list=[]

        for i in range(len(index1)):
            image_list.append(data_dir+str(index1[i])+'.jpg')

        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        # np.random.shuffle(temp)
    
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [round(float(i)) for i in label_list] 
        
        image = tf.cast(image_list, tf.string)
        label = tf.cast(label_list, tf.int32)

        # make an input queue
        input_queue = tf.train.slice_input_producer([image, label])
    
        label = input_queue[1]
        label = tf.cast(label, tf.int32)

        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)   #unit8,[h,w,3]            
                      
        image = tf.cast(image, tf.float32)

        image = tf.image.resize_image_with_crop_or_pad(image, img_width,  img_height) #裁剪到  207*207
        image = tf.image.resize_images(image, [80, 80],method=tf.image.ResizeMethod.BILINEAR) #图像放缩 到  69*69

    
        image = tf.cast(image, tf.float32)
        return image,label
    
def get_images_test(data_dir,label_dir):
    
#    img_width=180
#    img_width=200
    img_width=220
#    img_width=240
    img_height=img_width
#    img_depth = 3
    
    
    
    with tf.name_scope('input'):
        
        label=pd.read_csv(label_dir)
        index1=label['GalaxyID']

        label_list=label['class']
        label_list=list(label_list)
        image_list=[]

        for i in range(len(index1)):
            image_list.append(data_dir+str(index1[i])+'.jpg')

        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        # np.random.shuffle(temp)
    
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [round(float(i)) for i in label_list] 
        
        image = tf.cast(image_list, tf.string)
        label = tf.cast(label_list, tf.int32)

        # make an input queue
        input_queue = tf.train.slice_input_producer([image, label])
    
        label = input_queue[1]
        label = tf.cast(label, tf.int32)

        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)   #unit8,[h,w,3]            
                      
        image = tf.cast(image, tf.float32)

        image = tf.image.resize_image_with_crop_or_pad(image, img_width,  img_height) #裁剪到  207*207
        image = tf.image.resize_images(image, [80, 80],method=tf.image.ResizeMethod.BILINEAR) #图像放缩 到  69*69

    
        image = tf.cast(image, tf.float32)
        return image,label
        
    
  
    
#%% Reading training data

def read_galaxy11(data_dir,label_dir,batch_size):
    
    
    image,label=get_images(data_dir,label_dir)
        
    # data argumentation

    image = tf.random_crop(image, [64, 64, 3])# randomly crop the image size to 45 x 45
    image=tf.image.rot90(image,k=random.randint(0,3))
    image = tf.image.random_flip_left_right(image)
        
        
        
    image = tf.image.random_brightness(image, max_delta=63)
#    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
#    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image,lower=0.2,upper=1.8)    


    #归一化
    image = tf.image.per_image_standardization(image)   #substract off the mean and divide by the variance 


        
    images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 64,
                                    capacity = 20000,
                                    min_after_dequeue = 3000)
    images = tf.cast(images, tf.float32)    
    ## ONE-HOT      
    n_classes = 5
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
        
    return images, label_batch
#%%

def read_galaxy11_test(data_dir,label_dir, batch_size):
    """Read CIFAR10
    
    Args:
        data_dir: the directory of CIFAR10
        is_train: boolen
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    image,label=get_images_test(data_dir,label_dir)
        
    image = tf.image.resize_image_with_crop_or_pad(image, 64,64)

    #归一化
    image = tf.image.per_image_standardization(image)   #substract off the mean and divide by the variance 


        
    images, label_batch = tf.train.batch(
                                         [image, label],
                                         batch_size= batch_size,
                                         num_threads = 64,
                                         capacity= 2000)
    images = tf.cast(images, tf.float32)  
    ## ONE-HOT      
    n_classes = 5
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
        
    return images, label_batch




