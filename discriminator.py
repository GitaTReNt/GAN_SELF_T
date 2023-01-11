import tensorflow as tf
import keras
from keras import layers
def discriminator_model():
    model=keras.Sequential()
    model.add(layers.Flatten())#图片是一个三维数据，要输入到全连接层之前，先使用flatten层压平为一维的
    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))#最后输出为0,1只需要一层
    return model
