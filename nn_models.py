from keras.models import Model,Sequential # basic class for specifying and training a neural network
from keras.layers import *


kernel_size = 4 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 16 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 32 # ...switching to 64 after the first pooling layer
conv_depth_3 = 64
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
num_rep=2


def deep_cnn2(image_shape, num_classes,num_rep=num_rep):
    
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(BatchNormalization())

    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    for _ in range(num_rep):
        model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    for _ in range(num_rep):
        model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # conv [128]
    model.add(Convolution2D(conv_depth_3, (kernel_size,kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_3, (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(num_classes,(1, 1)))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    
    '''
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam',
                  metrics=['accuracy']) 
	
    model.summary()
    '''
    return model




def deep_cnn2_regressor(image_shape, num_outputs):
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(BatchNormalization())

    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(drop_prob_1))


    # conv [128]
    model.add(Convolution2D(conv_depth_3, (kernel_size,kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(conv_depth_3, (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(num_outputs,(1, 1)))
    model.add(GlobalAveragePooling2D())
    
    
    '''
    model.compile(loss='mean_squared_error', # using the cross-entropy loss function
                  optimizer='adam',
                  metrics=['mae']) 

    model.summary()
    '''
    return model
