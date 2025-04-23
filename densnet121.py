
import tensorflow as tf
from tensorflow.keras import layers, models

def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        bn1 = layers.BatchNormalization()(x)
        relu1 = layers.ReLU()(bn1)
        conv1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same')(relu1)  # Expands features

        # 3×3 Convolution
        bn2 = layers.BatchNormalization()(conv1)
        relu2 = layers.ReLU()(bn2)
        conv2 = layers.Conv2D(growth_rate, (3, 3), padding='same')(relu2)  # Extracts features

        x = layers.Concatenate()([x, conv2])
    
    return x


def transition_layer(x, reduction):
    bn = layers.BatchNormalization()(x)
    relu = layers.ReLU()(bn)
    filters = int(x.shape[-1] * reduction)
    conv = layers.Conv2D(filters, (1, 1), padding='same')(relu)
    pool = layers.AveragePooling2D((2, 2), strides=2, padding='same')(conv)
    return pool

def DenseNet121(input_shape=(224, 224, 3), num_classes=1000, growth_rate=32, reduction=0.5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
    # Dense Blocks
    print(x.shape)
    x = dense_block(x, 6, growth_rate)
    print(x.shape)
    x = transition_layer(x, reduction)
    print(x.shape)
    x = dense_block(x, 12, growth_rate)
    x = transition_layer(x, reduction)
    x = dense_block(x, 24, growth_rate)
    x = transition_layer(x, reduction)
    x = dense_block(x, 16, growth_rate)
    
    # Classification Layer
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Create Model
model = DenseNet121(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
