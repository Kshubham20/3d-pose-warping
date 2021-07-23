# Function to create Encoder model

def create_resnet_encoder():
    
    inputs = Input(shape=(256, 256, 3))
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=64,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 2, 3, 3]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=64)
    
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=128,
               padding="same")(t)
    t = layers.Reshape((32, 32, 64, 2))(t)
    t = layers.Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1,  1))(t)

    num_blocks_list = [2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block3d(t, downsample=(j==0 and i!=0), filters=64)

    outputs = t 
  
    return (inputs,outputs)
    model1 = create_resnet_encoder()
