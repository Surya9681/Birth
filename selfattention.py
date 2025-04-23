import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax, Input, Reshape, Multiply, Layer
from tensorflow.keras.models import Model

class SelfAttentionLayer(Layer):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()

    def build(self, input_shape):
        self.channels = input_shape[-1]  

        self.query_conv = Conv2D(self.channels, kernel_size=1, padding="same", use_bias=False)
        self.key_conv = Conv2D(self.channels, kernel_size=1, padding="same", use_bias=False)
        self.value_conv = Conv2D(self.channels, kernel_size=1, padding="same", use_bias=False)
        self.softmax = Softmax(axis=-1)

    def call(self, x):
        bat6ch, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        q = self.query_conv(x)  
        k = self.key_conv(x)   
        v = self.value_conv(x)  

        q_reshaped = tf.reshape(q, [batch, H * W, C])
        k_reshaped = tf.reshape(k, [batch, H * W, C])
        v_reshaped = tf.reshape(v, [batch, H * W, C])

        attention_scores = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)  # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)
        attention_output = tf.matmul(attention_scores, v_reshaped)  
        attention_output = tf.reshape(attention_output, [batch, H, W, C])

        return Multiply()([x, attention_output])

if __name__ == "__main__":
    input_tensor = Input(shape=(32, 32, 64))  # (H, W, C)

    attention_output = SelfAttentionLayer()(input_tensor)

    model = Model(inputs=input_tensor, outputs=attention_output)

    model.summary()
  