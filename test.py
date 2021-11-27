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
import pandas as pd
import json
from tools.image_augment import PhotometricDistort,RandomCrop,RandomFlipImage
from Model.LSENet import backbone
from Model.LSENet import head

#-----------------------visulization detection results-----------------------
log_dir = 'logs/'
class_colors = [[0,0,0],[255,0,0],[255,255,255],[255,102,0],[0,255,255],[255,255,0],[153,255,0] 
                ,[255,153,0],[0,0,255],[102,0,255],[255,0,255],[139,35,35]]     

def month2season(num):
    season_label = np.zeros((1,4))
    
    if num>= 3 and num <= 5:
        season_label[0,0] = 1     #Spring
    elif num>=6 and num<=8:
        season_label[0,1] = 1     #Summer
    elif num>=9 and num<=11:
        season_label[0,2] = 1     #Fall
    else:
        season_label[0,3] = 1     #Winter
    return season_label

NCLASSES = 12
HEIGHT = 352
WIDTH = 352
down_sample_size = 8
batch_size = 1

# get the model
img_input = Input(shape=(HEIGHT,WIDTH , 3 ))
season_input = Input(shape=(12,))
feats = backbone(img_input,season_input,HEIGHT,WIDTH)
o = head(feats,NCLASSES, HEIGHT, WIDTH, down_sample_size,batch_size)
model = Model(inputs=[img_input,season_input],outputs= o)

#********************************************************************#
model.load_weights(log_dir + '71.75.h5')
#********************************************************************#

#get the test set
with open(r"dataset/test_list.txt","r") as f:
    lines = f.readlines()

save_path = 'img_out/'

for jpg in lines:
    img = Image.open("dataset/trainX/"+jpg[:12])
    old_img = copy.deepcopy(img)
    old_img = np.array(old_img)[:,:,0:3]
    old_img = Image.fromarray(np.uint8(old_img))
                                                        
    img = np.array(img)[:,:,0:3]

    img = img/255
    img = np.pad(img, ((0, 12), (0, 92), (0, 0)), 'constant', constant_values=0)
    
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    
    #seasonal feature input ,one-hot
    season_name = int(jpg[4:6])
    season_label = np.zeros((1,12,))
    season_label[0,int(season_name)-1] = 1
    #season_label = month2season(season_name)

    pr = model.predict([img,season_label])[0]

    pr = pr.reshape((HEIGHT,WIDTH,NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((HEIGHT, WIDTH,3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    
    seg_img = seg_img[:340,:260,:]
    seg_img = Image.fromarray(np.uint8(seg_img))

    image = Image.blend(old_img,seg_img,0.7)
    image.save(save_path + jpg[:12])
    
#-----------------------compute mIoU-----------------------

#confusion matrix
def fast_hist(a, b, n):#a is ground truth，shape(H×W,)；b is the predict label，shape(H×W,)；n is number of classes
    k = (a >= 0) & (a < n) 
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):#caculate mIoU
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
def compute_mIoU( lines, num_classes):
    """
    Compute IoU given the predicted colorized images and 
    """
    print('Num classes', num_classes)
    name_classes = ["background","front1","front2","front3","front4","front5","front6","front7","front8","front9","front10","front11"]
    hist = np.zeros((num_classes, num_classes))
    pix_acc_total = 0
    pix_acc_num = 0

    for jpg in lines:
        truth = np.array(Image.open('dataset/trainY/' + jpg[:12]))
        
        img = Image.open('dataset/trainX/' + jpg[:12]) 
        img = np.array(img)[:,:,:3]

        img = img/255
        img = np.pad(img, ((0, 12), (0, 92), (0, 0)), 'constant', constant_values=0)
    
        img = img.reshape(-1,HEIGHT,WIDTH,3)
        
        #seasonal featur input,one-hot
        season_name = int(jpg[4:6])
        season_label = np.zeros((1,12,))
        season_label[0,int(season_name)-1] = 1
        #season_label = month2season(season_name)

        pr = model.predict([img,season_label])[0]
        
        pr = pr.reshape((HEIGHT,WIDTH,NCLASSES)).argmax(axis=-1)
        pr_img = pr[:340,:260]
        
        pix_acc = np.sum((truth== pr_img).astype('int'))/(truth.shape[0]*truth.shape[1])
        pix_acc_total = pix_acc_total + pix_acc
        pix_acc_num = pix_acc_num + 1
        hist += fast_hist(truth.flatten(), pr_img.flatten(), num_classes)
    
    mIoUs = per_class_iu(hist)
    
    for ind_class in range(num_classes):#output mIoU
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    
    print('Pixel Acc = ' + str(np.around(100 * pix_acc_total/pix_acc_num, 2)) + '%')
    return mIoUs    

if __name__ == "__main__":
    log_dir = 'logs/'
    HEIGHT=352
    WIDTH= 352
    NCLASSES = 12
    compute_mIoU(lines, NCLASSES)
