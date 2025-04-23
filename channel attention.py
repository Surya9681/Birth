import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Multiply, Add, Input, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras.activations import sigmoid

def channel_attention_module(x, ratio=8):
   
    channel_dim = x.shape[-1] 
    avg_pool = GlobalAveragePooling2D()(x)

    avg_pool = Reshape((1, 1, channel_dim))(avg_pool)  

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel_dim))(max_pool)  
    
    shared_dense_1 = Dense(channel_dim // ratio, activation="relu", use_bias=False)
    shared_dense_2 = Dense(channel_dim, activation="sigmoid", use_bias=False)

    avg_pool = shared_dense_1(avg_pool)
    avg_pool = shared_dense_2(avg_pool)

    max_pool = shared_dense_1(max_pool)
    max_pool = shared_dense_2(max_pool)

    attention = Add()([avg_pool, max_pool])  

    attention = sigmoid(attention) 

    return Multiply()([x, attention])  

if __name__ == "__main__":
    input_tensor = Input(shape=(32, 32, 64))

    output_tensor = channel_attention_module(input_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.summary()
