#仅用于预览
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
from matplotlib import pyplot as plt
from config import pic_dir,WIDTH,HEIGHT,crop_rect,image_count

images = []
for pics in tqdm(os.listdir(pic_dir)[:image_count]):
    pic = Image.open(pic_dir + pics).crop(crop_rect)#用于裁剪图片，在使用时需要引入Image，使用Image中的open(file)方法可返回一个打开的图片，使用crop([x1,y1,x2,y2])可进行裁剪，裁剪为(x1y1->x2y2)的长方形。
    pic.thumbnail((WIDTH,HEIGHT),Image.ANTIALIAS)#Image.thumbnail()函数用于制作当前图片的缩略图;ANTIALIAS是抗锯齿功能
    images.append(np.uint8(pic))
#标准化
images = np.array(images) / 255#先变成nparray格式才能进入神经网络，255是为了正则化;rgb通道数是3，但是每个通道内是0-255
print(images.shape) #?
plt.figure(1, figsize=(10, 10))#指定长宽为10x10的画图输出
for i in range(25):
    plt.subplot(5, 5, i+1)#plt.subplot()函数用于直接指定划分方式和位置进行绘图。参数分别为 行数、列数、索引值
    plt.imshow(images[i])
    plt.axis('off')
plt.show()
