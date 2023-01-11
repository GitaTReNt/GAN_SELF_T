from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

pic_dir = 'data/img_align_celeba/img_align_celeba/'
image_count = 10000
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2
WIDTH = 128
HEIGHT = 128
crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
images = []
for pics in tqdm(os.listdir(pic_dir))[:image_count]:
    pic = Image.open(pic_dir + pics).crop(crop_rect)#用于裁剪图片，在使用时需要引入Image，使用Image中的open(file)方法可返回一个打开的图片，使用crop([x1,y1,x2,y2])可进行裁剪。
    pic.thumbnail((WIDTH,HEIGHT),Image.ANTIALIAS)#Image.thumbnail()函数用于制作当前图片的缩略图