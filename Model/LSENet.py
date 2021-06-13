from keras.models import *
from keras.layers import *
from keras.activations import *
import keras.backend as K
import keras
from PIL import Image
import numpy as np
import random
import copy
import os
import tensorflow as tf
import math

def conv_block(input_tensor,filters):
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_block_1(input_tensor,filters):
    x = Conv2D(filters, (1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def location_attention(input_tensor,in_channel,n_classes, input_height, input_width, down_sample_size,batch_size):
    a = AveragePooling2D(pool_size=(down_sample_size, down_sample_size))(input_tensor)
    
    a = conv_block_1(a,32)
    #a = Dropout(rate = 0.1)(a)
    
    #Lcation encoding
    #a = coor_cat(input_height//down_sample_size, input_width//down_sample_size,batch_size)(a)
    a = positionalencoding2d(input_height//down_sample_size, input_width//down_sample_size,batch_size)(a)
    
    a = conv_block_1(a,64) 
    a = conv_block_1(a,64)
    a = Conv2D( n_classes , (1, 1),padding='same',name='pos_attention_feature_num')(a)
    
    a = Activation('sigmoid', name='pos_attention_down')(a)
    
    #bilinear interpolation
    a = UpSampling2D(size = (down_sample_size, down_sample_size), interpolation='bilinear', name='pos_attention')(a)
    
    return a

class channel_wise_multiply(Layer):
    def __init__(self):
        super(channel_wise_multiply, self).__init__()
        
    def call(self, input_tensor):
        o, s = input_tensor
        #get the shape
        self.height = o.shape[1]
        self.width = o.shape[2]
        self.channel = o.shape[-1]
        
        s = tf.expand_dims(s,axis=1) #(bs,1,classes)
        s = tf.expand_dims(s,axis=1) #(bs,1,1,classes)
        
        x = tf.tile(s, [1,self.height,self.width,1])  #(bs,h,w,classes)
        #res = tf.multiply(o,x)
        res = x
        return res
    
    def compute_output_shape(self,input_shape):
        return (None,input_shape[0][1], input_shape[0][2], input_shape[0][-1])

def channel_supervision_unit(tensor_list, season_feature, out_channel):
    o, s = tensor_list
    s = GlobalAveragePooling2D()(s)
    s = Dense(int(out_channel/2), activation = 'relu')(s)
    #s = Dropout(rate = 0.1)(s)
    s = concatenate([s,season_feature],axis=-1)
    #s = Dense(2 * int(out_channel/16), activation = 'relu')(s)
    s = Dense(out_channel, activation = 'relu')(s)
    o = channel_wise_multiply()([o,s])
    return o
    
    
def backbone_encoder(input_tensor, season_feature, input_height=352 ,  input_width=288):

    #img_input = Input(shape=(input_height,input_width , 3 ))

    # 352,352,3 -> 176,176,64
    x = conv_block(input_tensor,64)
    x = conv_block(x, 64)
    a = channel_supervision_unit([x, input_tensor], season_feature, 64)
    x = Add()([a,x])
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    

    # 176,176,64 -> 88,88,128
    ori = x
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    a = channel_supervision_unit([x, ori], season_feature, 128)
    x = Add()([a,x])
    f2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 88,88,128 -> 44,44,256
    ori = x
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    a = channel_supervision_unit([x, ori], season_feature, 256)
    x = Add()([a,x])
    f3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 44,44,256 -> 22,22,512
    ori = x
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    a = channel_supervision_unit([x, ori], season_feature, 512)
    x = Add()([a,x])
    f4 = x 
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 22,22,512 -> 22,22,1024
    ori = x
    x = conv_block(x, 1024)
    x = conv_block(x, 1024)
    a = channel_supervision_unit([x, ori], season_feature, 1024)
    x = Add()([a,x])
    f5 = x 

    return [f1 , f2 , f3 , f4 , f5 ]

def backbone_decoder(f,season_feature):
    
    f1, f2, f3, f4, f5 = f
    p5 = f5
    
    # 22,22,512+1024 -> 44,44,512
    o = UpSampling2D((2,2))(p5)
    o = concatenate([o,f4],axis=-1)
    ori = o
    o = conv_block(o, 512)
    o = conv_block(o, 512)
    a = channel_supervision_unit([o, ori], season_feature, 512)
    p4 = Add()([a,o])
  
    # 44,44,512+256 -> 88,88,256
    o = UpSampling2D( (2,2))(p4)
    o = concatenate([o,f3],axis=-1)
    ori = o
    o = conv_block(o, 256)
    o = conv_block(o, 256)
    a = channel_supervision_unit([o, ori], season_feature, 256)
    p3 = Add()([a,o])
    
    # 88,88,256+128 -> 176,176,128
    o = UpSampling2D( (2,2))(p3)
    o =  concatenate([o,f2],axis=-1)
    ori = o
    o = conv_block(o, 128)
    o = conv_block(o, 128)
    a = channel_supervision_unit([o, ori], season_feature, 128)
    p2 = Add()([a,o])

    # 176,176,128+64 -> 352,352,64
    o = UpSampling2D( (2,2))(p2)
    o = concatenate([o,f1],axis=-1)
    ori = o
    o = conv_block(o, 64)
    o = conv_block(o, 64)
    a = channel_supervision_unit([o, ori], season_feature, 64)
    p1 = Add()([a,o])
    
    return [p1,p2,p3,p4,p5]

def backbone( inputs, season_feature, input_height=352, input_width=288 ):
    
    feat = backbone_encoder(input_tensor = inputs, season_feature = season_feature, input_height=input_height, input_width=input_width )
    o = backbone_decoder(feat, season_feature)
    
    return o

class coor_cat(Layer):
    def __init__(self,area_num_h,area_num_w,batch_size):
        self.area_num_h = area_num_h
        self.area_num_w = area_num_w
        self.batch_size = batch_size
        super(coor_cat, self).__init__()
        
    def call(self, input_tensor):
        y_range = tf.range(self.area_num_h, dtype=tf.float32)
        x_range = tf.range(self.area_num_w, dtype=tf.float32)
        
        y_range = y_range / (self.area_num_h - 1.0)
        x_range = x_range / (self.area_num_w - 1.0)
        
        x_range = x_range[tf.newaxis, :]   # [1, w]
        y_range = y_range[:, tf.newaxis]   # [h, 1]
        x = tf.tile(x_range, [self.area_num_h, 1])     # [h, w]
        y = tf.tile(y_range, [1, self.area_num_w])     # [h, w]
        
        x = x[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
        y = y[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
        x = tf.tile(x, [self.batch_size, 1, 1, 1])   # [N, h, w, 1]
        y = tf.tile(y, [self.batch_size, 1, 1, 1])   # [N, h, w, 1]
        
        res = tf.concat([input_tensor, x, y], axis=-1)   # [N, h, w, c+2]
        return res
    
    def compute_output_shape(self, input_shape):
        return (None,input_shape[1], input_shape[2], input_shape[-1]+2 )
    
class positionalencoding2d(Layer): 
    def __init__(self,height,width,batch_size):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        super(positionalencoding2d, self).__init__()

    def call(self, input_tensor):
        self.channel_num = input_tensor.shape[-1]
        d_model = int(self.channel_num)
        height = self.height
        width = self.width
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
        pe = np.zeros([height, width, d_model])
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        
        div_term = np.exp( np.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = np.expand_dims(np.arange(0., width),axis=1)
        pos_h = np.expand_dims(np.arange(0., height),axis=1)
    
        pe[ :, :, 0:d_model:2] = np.expand_dims(np.sin(pos_w * div_term), axis=0).repeat(height,axis = 0)
        pe[ :, :, 1:d_model:2] = np.expand_dims(np.cos(pos_w * div_term), axis=0).repeat(height,axis = 0)
        pe[:, :, d_model::2] = np.expand_dims(np.sin(pos_h * div_term), axis=1).repeat(width,axis = 1)
        pe[ :, :, d_model + 1::2] = np.expand_dims(np.cos(pos_h * div_term), axis=1).repeat(width,axis =1)
    
        #add a dimension for sum operation
        pe = np.expand_dims(pe,axis=0).repeat(self.batch_size,axis=0)
        
        res = tf.add(input_tensor,tf.convert_to_tensor(pe, np.float32))
        return res

    def compute_output_shape(self, input_shape):
        return (None,input_shape[1], input_shape[2], input_shape[-1] )
    
def head(feats, n_classes, input_height, input_width, down_sample_size, batch_size):
    p1, p2, p3, p4, p5 = feats
    
    o = Conv2D( n_classes, (1, 1),padding='same',name='1')(p1)
    
    a = location_attention(p1,64, n_classes, input_height, input_width, down_sample_size,batch_size)

    o = Softmax()(o)
    a = Multiply(name='2')([o,a])
    o = Add()([o,a])
    o = Reshape((input_height*input_width, -1))(o)
    
    return o
