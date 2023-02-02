from tensorflow.keras import layers
import tensorflow_addons as tfa




class Conv_block(layers.Layer):
    def __init__(self, out_channels, kernel_size=(3, 3), dilation_rate=1):
        super(Conv_block, self).__init__()
        self.conv = layers.Conv2D(
            out_channels, kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=True,
        )
        self.GN = tfa.layers.GroupNormalization(groups=32)
        self.LeakyRelu = layers.LeakyReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.GN(x)
        x = self.LeakyRelu(x)
        return x


class SA_Block(layers.Layer):
    def __init__(self, out_channels):
        super(SA_Block, self).__init__()  # SA_Block or Conv_block
        self.conv1 = Conv_block(out_channels=out_channels, kernel_size=3)
        self.conv2 = Conv_block(out_channels=out_channels, kernel_size=3)

        self.out_channels = out_channels
        self.AVGpool = layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        self.Upsample = layers.UpSampling2D(size=(2, 2))
        self.multiply = layers.Multiply()
        self.add = layers.Add()

    def call(self, input):
        x_res = self.conv1(input)
        y_res = self.conv2(x_res)

        x_atten = self.AVGpool(input)
        x_atten = self.conv1(x_atten)
        x_attne = self.conv2(x_atten)
        y_attention = self.Upsample(x_atten)
        out_mul = self.multiply([y_attention, y_res])
        out = self.add([out_mul, y_attention])

        return out


class Downsample(layers.Layer):
    def __init__(self, out_channels):
        super(Downsample, self).__init__()
        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.avgpool = layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        self.concat = layers.Concatenate(axis=-1)
        self.conv_down = Conv_block(out_channels, kernel_size=(1, 1))

    def call(self, input):
        x_max = self.maxpool(input)
        x_avg = self.avgpool(input)
        x = self.concat([x_max, x_avg])
        x = self.conv_down(x)
        return x


class Upsample(layers.Layer): 
    def __init__(self, out_channels):
        super(Upsample, self).__init__()
        self.upsampling = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv = Conv_block(out_channels, kernel_size=(1, 1))
    def call(self, input):
        x = self.upsampling(input)
        x = self.conv(x)
        return x


class DASPP(layers.Layer):
    def __init__(self, out_channels):
        super(DASPP, self).__init__()
        self.conv_block_pool = Conv_block(out_channels, kernel_size=(1, 1), dilation_rate=1)
        self.conv_block1 = Conv_block(out_channels, kernel_size=(1, 1), dilation_rate=1)
        self.conv_block6 = Conv_block(out_channels, kernel_size=(3, 3), dilation_rate=6)
        self.conv_block12 = Conv_block(out_channels, kernel_size=(3, 3), dilation_rate=12)
        self.conv_block18 = Conv_block(out_channels, kernel_size=(3, 3), dilation_rate=18)
        self.avgpool = layers.AveragePooling2D
        self.upsampling = layers.UpSampling2D
        self.concat = layers.Concatenate(axis=-1)
        self.conv_block_out = Conv_block(out_channels, kernel_size=(1, 1), dilation_rate=1)
    def call(self, input):
        dims = input.shape
        x_pool = self.avgpool(pool_size=(dims[-3], dims[-2]))(input)
        x_pool = self.conv_block_pool(x_pool)
        out_pool = self.upsampling(size=(dims[-3], dims[-2]))(x_pool)
        out_1 = self.conv_block1(input)
        out_6 = self.conv_block6(input)
        x_12 = self.concat([input, out_6])
        out_12 = self.conv_block12(x_12)
        x_18 = self.concat([input, x_12, out_12])
        out_18 = self.conv_block18(x_18)
        x = self.concat([out_pool, out_1, out_6, out_12, out_18])
        output = self.conv_block_out(x)
        return output

class MyModel(keras.Model):
  def __init__(self, num_classes):
    super(MyModel, self).__init__()
    self.concat = layers.Concatenate(axis=-1)
    self.Up = Upsample   
    self.Down = Downsample
    self.DASPP = DASPP
    self.SA = SA_Block
    self.conv = layers.Conv2D
    self.num_classes = num_classes

  def call(self, input):
    x1 = SA_Block(32)(input)
    x2 = self.Down(32)(x1)
    print("Stage1", x2.shape)

    x3 = self.SA(64)(x2)
    x4 = self.Down(64)(x3)
    print("Stage2", x4.shape)

    x5 = self.SA(128)(x4)
    x6 = self.Down(128)(x5)
    print("Stage3", x6.shape)

    x7 = self.SA(256)(x6)
    x8 = self.Down(256)(x7)
    print("Stage4", x8.shape)

    x9 = self.SA(512)(x8)
    print("Stage5", x9.shape)
    
    x10 = self.DASPP(256)(x9)
    print("Stage6", x10.shape)

    x11 = self.Up(256)(x10)
    print("Stage7", x11.shape)
    x12 = self.concat([x11, x7])
    print("Stage7_1", x12.shape)
    x13 = self.SA(256)(x12)
    print("Stage8", x13.shape)

    x14 = self.Up(128)(x13)
    print("Stage9", x14.shape)
    x15 = self.concat([x14, x5])
    print("Stage9_1", x15.shape)
    x16 = self.SA(128)(x15)
    print("Stage10", x16.shape)

    x17 = self.Up(64)(x16)
    print("Stage11", x17.shape)
    x18 = self.concat([x17, x3])
    print("Stage11_1", x18.shape)
    x19 = self.SA(64)(x18)
    print("Stage12", x19.shape)

    x20 = self.Up(32)(x19)
    print("Stage13", x20.shape)
    x21 = self.concat([x20, x1])
    print("Stage13_1", x21.shape)
    x22 = self.SA(32)(x21)
    print("Stage14", x22.shape)

    if self.num_classes == 2:
      activation = 'sigmoid'

    else:
      activation = 'softmax'
    
    out = self.conv(filters=self.num_classes, kernel_size=3, 
                                padding="same", use_bias=True, 
                                activation=activation)(x22)
                                
    return out