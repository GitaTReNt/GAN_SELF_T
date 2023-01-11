import tensorflow as tf
import keras

keras=tf.keras
layers=keras.layers
def generator_model():
    model=keras.Sequential()#
    model.add(layers.Dense(256,use_bias=False))#输入形状100是我输入噪声的形状,生成器一般都不使用BIAS input_shape=(100,),
    model.add(layers.BatchNormalization())#全连接层--批标准化--激活
    model.add(layers.LeakyReLU())#GAN中一般使用LeakyRelu函数来激活
    model.add(layers.Dense(512,use_bias=False))#生成器一般都不使用BIAS
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())#全连接层--批标准化--激活
    model.add(layers.Dense(28*28*1,use_bias=False,activation='tanh'))#生成能够调整成我们想要的图片形状的向量 生成器最后没激活函数，生成的图片会有很明显的奇怪噪点。
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((28,28,1)))#这里进行修改向量的形状，可以直接使用layers的reshape
    return model
#在多层CNN里，BN放在卷积层之后，激活和池化之前，以LeNet5为例：
#由于经过Batch Norm处理时，通过训练β参数，进对线性变换的结果做了合适的平移，bias项可以忽略不用。
#对卷积处理(cross-correlation)：zd,i,j=∑ch=0CH−1∑r=0R−1∑c=0C−1wd,ch,r,cxch,i+r,j+c+bd\displaystyle z_{d,i,j}=\sum \limits_{ch=0}^{CH-1}\sum\limits_{r=0}^{R-1}\sum\limits_{c=0}^{C-1}w_{d,ch,r,c}x_{ch,i+r,j+c}+b_d  \displaystyle z_{d,i,j}=\sum \limits_{ch=0}^{CH-1}\sum\limits_{r=0}^{R-1}\sum\limits_{c=0}^{C-1}w_{d,ch,r,c}x_{ch,i+r,j+c}+b_d
# 如果置于Batch Norm层之前，同样可以忽略bias项。
'''由于BN是对数据进行规范化操作，因此理论上，BN可以在网络中的任意位置使用。
在实际应用中，通常是两种做法，一种是在激活函数前使用，一种是在激活函数后使用。
在激活函数前使用时，BN后的数据可以直接作为激活函数的输入，缓解激活函数的输出落入梯度饱和区。
在激活函数后使用时，相当于BN对整个隐层的输出进行操作，使下一层隐层的输入数据在相对稳定的分布。
当激活函数是relu时，需避免在激活函数后使用BN，因为relu激活函数会对信号过滤，将小于0的信号置0，
导致一些神经元失活，对BN结果造成不稳定，进而影响模型收敛的稳定性。

'
'''