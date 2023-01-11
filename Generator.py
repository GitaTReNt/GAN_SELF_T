from keras import Input
from keras.models import Model
from config import LATENT_DIM,CHANNELS
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose

def generator_model():
    gen_input = Input(shape = (LATENT_DIM,))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)#卷积
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)#反卷积操作
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)#最后都用tanh激活，体现全局的特征

    generator = Model(gen_input, x)

    return generator
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