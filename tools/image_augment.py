import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

def PhotometricDistort(image_origin):
    
    image = image_origin.astype(np.float32)
    
    if np.random.randint(2):
        # RandomBrightness
        if np.random.randint(2):     #np.random.randint(2)生成0或1的随机数，也就是说 下面的操作有50%的概率会执行
            delta = 32
            delta = np.random.uniform(-delta, delta)  #随机来个亮度，实际就是生成一个-32到32之间的值，然后叠加到原图片上
            image += delta

        state = np.random.randint(2)
        if state == 0:           #50%的概率执行下述操作
            if np.random.randint(2): #又50%概率执行下述操作，也就是 1/4概率
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)  
                image *= alpha                 #图片整体像素放大或缩小 0.5-1.5 倍

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)    #将图片转换为HSV
        if np.random.randint(2):                   #1/2概率执行HSV图片修改操作; HSV: 色调（H），饱和度（S），明度（V）
            lower = 0.5
            upper = 1.5
            image[:, :, 1] *= np.random.uniform(lower, upper)  #改饱和度

        if np.random.randint(2):
            delta = 18.0
            image[:, :, 0] += np.random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0    #改色调

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if state == 1:        #保证这个事件有且只会发生一次
            if np.random.randint(2): #50%概率对图片的整体像素进行放大或缩小 0.5-1.5 倍，
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha
    
    return image

def RandomCrop(image_origin, mask_origin, h = 352, w = 352,cut_label=[1], 
               allow_no_crop=True, num_attempts = 10, scaling=[.4, 1.]):
    
    image = image_origin
    masks = mask_origin
    cut_label = list(cut_label)     #1/2概率被裁减
    
    if allow_no_crop:
        cut_label.append('no_crop')
    np.random.shuffle(cut_label)
        
    cut = cut_label[0]
    if cut == 'no_crop':
        #填边缘,往下和右添加
        padding_h = h - image.shape[0]
        padding_w = w - image.shape[1]
        image = np.pad(image, ((0, padding_h), (0, padding_w), (0, 0)), 'constant', constant_values=0)
        masks = np.pad(masks, ((0, padding_h), (0, padding_w)), 'constant', constant_values=0)
        return image, masks    #每次是随机的，如果随到no_crop,则不进行裁剪

    found = False
    for i in range(num_attempts):   #进行num_attempts次尝试
        scale = np.random.uniform(*scaling) #随机生成目标图像和原始图像的比例大小
                
        crop_h = int(h * scale )  #裁剪完毕的目标的高
        crop_w = int(w * scale)  #裁剪完毕的目标的宽
        crop_y = np.random.randint(0, h - crop_h)   #剩下的高度里选一个随机数，作为初始y
        crop_x = np.random.randint(0, w - crop_w)   #剩下的宽度里选一个随机数，作为初始x
        crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h] #裁剪框的左上角和右下角坐标
        
        image_tmp =  image[ crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        mask_tmp = masks[ crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        
        if np.any(mask_tmp!=0):   #截取的部分存在锋面
            found = True
            break

    if found:
        #填边缘
        padding_h = h - image_tmp.shape[0]
        padding_w = w - image_tmp.shape[1]
        image = np.pad(image_tmp, ((0, padding_h), (0, padding_w), (0, 0)), 'constant', constant_values=0)
        masks = np.pad(mask_tmp, ((0, padding_h), (0, padding_w)), 'constant', constant_values=0)
        return image, masks
    
    #填边缘
    padding_h = h - image.shape[0]
    padding_w = w - image.shape[1]
    image = np.pad(image, ((0, padding_h), (0, padding_w), (0, 0)), 'constant', constant_values=0)
    masks = np.pad(masks, ((0, padding_h), (0, padding_w)), 'constant', constant_values=0)
    return image, masks

def RandomFlipImage(image, mask):
    image = Image.fromarray(np.uint8(image))
    mask = Image.fromarray(np.uint8(mask))
    
    #1/2概率翻转
    if np.random.randint(2):
        flip_label = np.random.randint(4)
        #翻转分为4种，90，180，270，左右镜像
        if flip_label==0:
            image = image.rotate(90)
            mask = mask.rotate(90)
        elif flip_label==1:
            image = image.rotate(180)
            mask = mask.rotate(180)
        elif flip_label==2:
            image = image.rotate(270)
            mask = mask.rotate(270)
        else:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
    return np.array(image), np.array(mask) 
