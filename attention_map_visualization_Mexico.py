from keras.models import *
from keras.layers import *
from keras.activations import *
import keras.backend as K
import keras
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import os
import copy
import random
import matplotlib.pyplot as plt
from Model.LSENet import backbone
from Model.LSENet import head
import matplotlib

log_dir = 'logs/'
class_colors = [[0,0,0],[255,0,0],[255,255,255],[255,102,0],[0,255,255],[255,255,0],[153,255,0] # 分别为 黑色， 红色，白红，橙色，
                ,[255,153,0],[0,0,255],[102,0,255],[255,0,255],[139,35,35]]     # 蓝绿，黄色，黄绿，橙黄，蓝色，蓝紫，紫色，棕色
class_pick = 1
NCLASSES = 6
HEIGHT = 384
WIDTH = 384
down_sample_size = 24
batch_size = 1
save_path = 'attention_map'

# get the model
img_input = Input(shape=(HEIGHT,WIDTH , 3 ))
season_input = Input(shape=(12,))
feats = backbone(img_input,season_input,HEIGHT,WIDTH)
o = head(feats,NCLASSES, HEIGHT, WIDTH, down_sample_size,batch_size)
model = Model(inputs=[img_input,season_input],outputs= o)

#********************************************************************#
model.load_weights(log_dir + '78.31.h5')
#********************************************************************#

# source image
# jpg = '20111023.png'
jpg = '20101214.png'

img = Image.open("dataset_mexico/trainX/"+jpg[:12])
old_img = copy.deepcopy(img)
old_img = np.array(old_img)[:,:,0:3]
old_img = Image.fromarray(np.uint8(old_img))
                                                        
img = np.array(img)[:,:,0:3]

img = img/255
img = np.pad(img, ((0, 120), (0, 20), (0, 0)), 'constant', constant_values=0)
    
img = img.reshape(-1,HEIGHT,WIDTH,3)
    
#season输入,one-hot
season_name = int(jpg[4:6])
season_label = np.zeros((1,12,))
season_label[0,int(season_name)-1] = 1
#season_label = month2season(season_name)

pr = model.predict([img,season_label])[0]

#-----------------------get output of attention layer-----------------------

# layer_name = 'pos_attention_down'
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
# pr = intermediate_layer_model.predict([img,season_label])[0]
# attention_down_ori = pr
# attention_down = pr[:31,:24]

layer_name = 'pos_attention'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
pr = intermediate_layer_model.predict([img,season_label])[0]
attention = pr[:264,:364]

#-----------------------find edge of the ocean front------------------------------
img = Image.open(r"dataset_mexico/trainY" + '/' + jpg)
img = np.array(img)
edge_img = np.zeros((264,364))
edge_labels = np.zeros((264,364,NCLASSES))

# latitude scan
for i in range(264):
    for j in range(1,6):
        index = np.argwhere(img[i] == j) # Get all of the indexes in target class
        if index.shape != (0,1):         # If there is a index
            edge_img_row = np.zeros((20,364)) # default maximum 20 consecutive arrays, each array up to 364 pixels
            seq_num = 0 
            if index.shape[0]==1:
                edge_img[i,int(index[0,0])] = j
            else:
                index_temp = np.squeeze(index)
                for k in range(index_temp.shape[0]-1):
                    diff = index_temp[k+1] - index_temp[k]
                    if diff == 1:
                            edge_img_row[seq_num, k] = index_temp[k]      
                            edge_img_row[seq_num, k+1] = index_temp[k+1]  
                    else:
                            edge_img_row[seq_num, k] = index_temp[k]
                            seq_num = seq_num + 1
                            edge_img_row[seq_num, k+1] = index_temp[k+1]
            
                for l in range(20):
                    a = edge_img_row[l]
                    if np.any(a!=0): 
                        minval = np.min(a[np.nonzero(a)])
                        maxval = np.max(a[np.nonzero(a)])
                    
                        #fill number
                        edge_img[i,int(minval)] = j
                        edge_img[i,int(maxval)] = j
                        
#longitudinal scanning
for i in range(364):
    for j in range(1,6):
        index = np.argwhere(img[:,i] == j)    #Get all of the indexes in target class
        if index.shape != (0,1):              #If there is a index
            edge_img_row = np.zeros((20,264)) #default maximum 20 consecutive arrays, each array up to 264 pixels
            seq_num = 0 
            if index.shape[0]==1:
                edge_img[int(index[0,0]),i] = j
            else:
                index_temp = np.squeeze(index)
                for k in range(index_temp.shape[0]-1):
                    diff = index_temp[k+1] - index_temp[k]
                    if diff == 1:
                            edge_img_row[seq_num, k] = index_temp[k]      
                            edge_img_row[seq_num, k+1] = index_temp[k+1]  
                    else:
                            edge_img_row[seq_num, k] = index_temp[k]
                            seq_num = seq_num + 1
                            edge_img_row[seq_num, k+1] = index_temp[k+1]
            
                for l in range(20):
                    a = edge_img_row[l]
                    if np.any(a!=0): 
                        minval = np.min(a[np.nonzero(a)])
                        maxval = np.max(a[np.nonzero(a)])
                    
                        #fill numbers
                        edge_img[int(minval),i] = j
                        edge_img[int(maxval),i] = j
                        
for c in range(NCLASSES):
    edge_labels[: , : , c ] = (edge_img[:,:] == c ).astype(int)

edge = edge_labels[:,:,class_pick]

if class_pick!= 0:
    edge = edge_labels[:,:,class_pick]

    a = np.argwhere(edge==1)
    x = a[:,0]
    y = a[:,1]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=(11,8))     

    ax = fig.add_subplot(1,1,1)

    new_ticks_x = [6,66,126,186,246,306,355]
    ax.set_xticklabels(['98.5°W','95.5°W','92.5°W','89.5°W','86.5°W','83.5°W','80.5°W'])
    plt.xticks(new_ticks_x)

    new_ticks_y = [9,49,89,129,169,209,249]
    ax.set_yticklabels(['30.5°N','28.5°N','26.5°N','24.5°N','22.5°N','20.5°N','18.5°N'])
    plt.yticks(new_ticks_y)

    plt.yticks(fontproperties = 'Times New Roman', size = 11)
    plt.xticks(fontproperties = 'Times New Roman', size = 11)

    con = plt.pcolormesh(attention[:,:,class_pick],norm=norm, cmap = 'coolwarm')
    ax = plt.gca()
    plt.scatter(y, x, s=4.5, c='k')
    ax.invert_yaxis()

    # plt.imshow()
    plt.savefig(save_path +'/'+ jpg[:8] + '_class' +str(class_pick) + '.png', dpi = 300, bbox_inches = 'tight',pad_inches = 0)
else:
    edge = edge_img
    a = np.argwhere(edge!=0)
    x = a[:,0]
    y = a[:,1]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(11,8))      

    ax = fig.add_subplot(1,1,1)

    new_ticks_x = [6,66,126,186,246,306,355]
    ax.set_xticklabels(['98.5°W','95.5°W','92.5°W','89.5°W','86.5°W','83.5°W','80.5°W'])
    plt.xticks(new_ticks_x)

    new_ticks_y = [9,49,89,129,169,209,249]
    ax.set_yticklabels(['30.5°N','28.5°N','26.5°N','24.5°N','22.5°N','20.5°N','18.5°N'])
    plt.yticks(new_ticks_y)

    plt.yticks(fontproperties = 'Times New Roman', size = 11)
    plt.xticks(fontproperties = 'Times New Roman', size = 11)

    con = plt.pcolormesh(attention[:,:,class_pick],norm=norm, cmap = 'coolwarm')
    ax = plt.gca()
    plt.scatter(y, x, s=4.5, c='k')

    ax.invert_yaxis()

    # plt.imshow()
    plt.savefig(save_path +'/'+ jpg[:8] + '_class' +str(class_pick) + '.png', dpi = 300, bbox_inches = 'tight',pad_inches = 0)
    
    #color bar plot
    #cbar = plt.colorbar(con,fraction=0.06, pad=0.06)
    #for l in cbar.ax.yaxis.get_ticklabels():
     #   l.set_family('Times New Roman')
   # cbar.ax.tick_params(labelsize=12)  
    #plt.savefig(save_path +'/'+ jpg[:8] + '_class_color_bar' +str(class_pick) + '.png', dpi = 300, bbox_inches = 'tight',pad_inches = 0)
