import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2


def mobilenet_mod(lambd, dim, output_neurons, output_activation):    
    base_model = tf.keras.applications.MobileNet(input_shape = dim, weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(128, kernel_regularizer=l2(lambd), bias_regularizer=l2(lambd))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(output_neurons, activation = output_activation)(x)  

    model = Model(inputs = base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = True

    return model