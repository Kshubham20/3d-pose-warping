!pip install tensorflow-addons
from tensorflow import Tensor 
from tensorflow.keras.layers import Input, Conv3D, ReLU, Add, AveragePooling3D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

from tensorflow_addons.layers import GroupNormalization

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

#Function for 2D Residual Block

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def residual_block_decode(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = layers.Conv2DTranspose(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out
    #Function for 3D Residual Block

def relu_bn3d(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = GroupNormalization()(relu)
    return bn

def residual_block3d(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv3D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn3d(y)
    y = Conv3D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = Conv3D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn3d(out)
    return out
