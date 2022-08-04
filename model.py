import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

#The Location Encoding layer model that uses the relative orientation of edges and features to learn invariant shape features
class LocationEncoding(tf.keras.layers.Layer):
  def __init__(self, filters1, filters2):
    super(LocationEncoding, self).__init__()
    self.filters1 = filters1
    self.filters2 = filters2
    #self.positions=positions

    self.conv1 = keras.layers.Conv2D(self.filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name='conv2a',padding="same")
    self.wp = keras.layers.Conv2D(self.filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name='conv2b',padding="same")
    self.conv2 = layers.Conv2D(self.filters2, (3, 3),
                      kernel_initializer='he_normal',
                      name='conv2c',groups=8,padding="same")
    self.conv3 = layers.Conv2D(self.filters2, (3, 3),
                      kernel_initializer='he_normal',
                      name='conv2d',activation='softmax',kernel_regularizer=keras.regularizers.L2(0.0001),padding="same")
    self.conv4 = layers.Conv2D(self.filters2, (1, 1),
                      kernel_initializer='he_normal',
                      name='conv2d',activation='softmax',kernel_regularizer=keras.regularizers.L1(0.0001),padding="same")
    self.conv5 = layers.Conv2D(self.filters2, (1, 1),
                      kernel_initializer='he_normal',
                      name='conv2d',activation='elu',kernel_regularizer=keras.regularizers.L2(0.0001),padding="same")
    self.convl = layers.Conv2D(self.filters2, (1, 1),
                      kernel_initializer='he_normal',
                      name='conv2d',activation='relu',kernel_regularizer=keras.regularizers.L2(0.0001),padding="same")
    
    self.bn = layers.BatchNormalization(name='bn')
    self.bn1 = layers.BatchNormalization(name='bn1')
    self.bn2 = layers.LayerNormalization(name='bn2')
    self.bn3 = layers.LayerNormalization(name='bn3')
    self.bn4 = layers.LayerNormalization(name='bn4')

    #self.dense = tf.keras.layers.Dense(d_model)

  def call(self, input_tensor, image):

    x = self.conv1(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    positions = image[:,:,:,-2:]
    positions = tf.tile(positions,[1,1,1,self.filters1])
    positions = layers.AveragePooling2D(pool_size=(image.shape[1]/input_tensor.shape[1],image.shape[2]/input_tensor.shape[2]),strides=(image.shape[1]/input_tensor.shape[1],image.shape[2]/input_tensor.shape[2]))(positions)
    #x1loc = tf.where(tf.reduce_any(x>0),positions,0.0)
    x1 = tf.pad(layers.MaxPooling2D(pool_size=(2,2),padding="same")(x),paddings=[[0,0],[int(x.shape[1]/4),int(x.shape[1]/4)],[int(x.shape[1]/4),int(x.shape[1]/4)],[0,0]])
    x1loc = tf.where(tf.tile(x,[1,1,1,2])>0,positions,0.0)
    #x3 = tf.pad(layers.MaxPooling2D(pool_size=(8,8),padding="same")(x),paddings=[[0,0],[int(x.shape[1]/2),int(x.shape[1]/2)],[int(x.shape[1]/2),int(x.shape[1]/2)],[0,0]])
    x3 = layers.MaxPooling2D(pool_size=(5,5),strides=(1,1),padding="same")(x)
    #x3 = tf.image.resize_with_crop_or_pad(x3,x.shape[1],x.shape[2])
    xa = tf.math.argmax(x,1)
    xa = tf.math.argmax(xa,2)

    x1loc = tf.gather(x1loc,xa,axis=2,batch_dims=1)
    locs = custom_comp(x1loc,x1loc)
    #xpose = self.conv3(locs)
    x = self.bn(x)
    x = layers.Add()([x,self.conv5(x3)])
    #x = self.bn4(x)
    xpose = self.conv4(locs) 
    x = layers.Add()([x,xpose])
    x = self.bn3(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x
  
  
