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
import random
from tools.image_augment import PhotometricDistort,RandomCrop,RandomFlipImage
from Model.LSENet import backbone
from Model.LSENet import head

NCLASSES = 12
HEIGHT = 352
WIDTH = 352
down_sample_size = 8
batch_size = 4

def month2season(num):
    season_label = np.zeros((4,))
    
    if num>= 3 and num <= 5:
        season_label[0] = 1     #Spring
    elif num>=6 and num<=8:
        season_label[1] = 1     #Summer
    elif num>=9 and num<=11:
        season_label[2] = 1     #Fall
    else:
        season_label[3] = 1     #Winter
    
    return season_label

def generate_train_arrays_from_file(lines,batch_size):

    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        season_X = []

        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
                
            #trainX    
            name = lines[i]
            img = Image.open(r"dataset/trainX" + '/' + name)
            img = np.array(img) 
            img = img[:,:,0:3]
            
            #seasonal feature input,one-hot
            season_name = int(name[4:6])
            season_label = np.zeros((12,))
            season_label[int(season_name)-1] = 1
            #season_label = month2season(season_name)
            season_X.append(season_label)
            
            #trainY
            name = lines[i]
            mask = Image.open(r"dataset/trainY" + '/' + name)
            mask = np.array(mask)
            
            img = PhotometricDistort(img)
            img, mask = RandomCrop(img,mask)
            img, masks = RandomFlipImage(img,mask)
            image = img/255
            X_train.append(image)
            
            seg_labels = np.zeros((HEIGHT,WIDTH,NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (masks[:,:] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            i = (i+1) % n
        yield ([np.array(X_train),np.array(season_X)],np.array(Y_train))        

def generate_valid_arrays_from_file(lines,batch_size):
    
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        season_X = []

        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
                
            #trainX    
            name = lines[i]
            img = Image.open(r"dataset/trainX" + '/' + name)
            img = np.pad(np.array(img), ((0, 12), (0, 92), (0, 0)), 'constant', constant_values=0)
            img = img[:,:,0:3]
            img = img/255
            X_train.append(img)
            
            #seasonal feature input,one-hot
            season_name = int(name[4:6])
            season_label = np.zeros((12,))
            season_label[int(season_name)-1] = 1
            #season_label = month2season(season_name)
            season_X.append(season_label)
            
            #trainY
            name = lines[i]
            img = Image.open(r"dataset/trainY" + '/' + name)
            img = np.pad(np.array(img), ((0, 12), (0, 92)), 'constant', constant_values=0)
            seg_labels = np.zeros((HEIGHT,WIDTH,NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            i = (i+1) % n
        yield ([np.array(X_train),np.array(season_X)],np.array(Y_train)) 
        
        
def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss

if __name__ == "__main__":
    log_dir = "logs/"
    cross_valid = 2018
    
    # build model
    img_input = Input(shape=(HEIGHT,WIDTH , 3 ))
    season_input = Input(shape=(12,))
    feats = backbone(img_input,season_input,HEIGHT,WIDTH)
    o = head(feats,NCLASSES, HEIGHT, WIDTH, down_sample_size,batch_size)
    model = Model(inputs=[img_input,season_input],outputs= o)
    # model.summary()
    
    lines = os.listdir("dataset/trainX/")
    sort_lines = []
    for names in lines:
        sort_lines.append(names[:8])
    sort_lines = np.sort(np.array(sort_lines))
    
    line=[]
    for names in sort_lines:
        line.append(str(names)+'.png')

    data_2015 = line[:365]
    data_2016 = line[365:731]
    data_2017 = line[731:1096]
    data_2018 = line[1096:]
    
    train = []
    test = []

    if cross_valid == 2015:
        test.extend(data_2015)
        train.extend(data_2016)
        train.extend(data_2017)
        train.extend(data_2018)
    
    elif cross_valid == 2016:
        test.extend(data_2016)
        train.extend(data_2015)
        train.extend(data_2017)
        train.extend(data_2018)
    
    elif cross_valid == 2017:
        test.extend(data_2017)
        train.extend(data_2018)
        train.extend(data_2016)
        train.extend(data_2015)
    
    else:
        test.extend(data_2018)
        train.extend(data_2015)
        train.extend(data_2016)
        train.extend(data_2017)
        
    #Generate training set   
    random.shuffle(train)
    num_val = int(len(train)*0.1)
    num_train = len(train) - num_val

    #Save test set to txt file
    file_test=open('dataset/test_list.txt',mode='w')
    for names in test:
        file_test.writelines(names+'\n')
    file_test.close()
    
    # save weights in each 2 generation
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=2
                                )
    
    # decay strage
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )

    # loss
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # train 
    model.fit_generator(generate_train_arrays_from_file(train[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_valid_arrays_from_file(train[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=80,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir+'last1.h5')
