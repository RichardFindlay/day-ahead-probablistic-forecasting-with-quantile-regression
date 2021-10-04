import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Softmax, Bidirectional 
from tensorflow.keras.layers import Activation, concatenate


#one-step temporal Atttention
class attention(tf.keras.layers.Layer):

	def __init__(self, hidden_units, **kwargs):
		super(attention, self).__init__(hidden_units)
		self.hidden_units = hidden_units
		super(attention, self).__init__(**kwargs)

	def build(self, input_shape):

		self.conv1d_1 = Conv1D(self.hidden_units, kernel_size=1, strides=1, padding='same', activation='relu')
		self.conv1d_2 = Conv1D(self.hidden_units, kernel_size=1, strides=1, padding='same', activation='relu')
		self.conv1d_3 = Conv1D(1, kernel_size=1, strides=1, padding='same', activation='relu')

		self.tanh = tf.keras.layers.Activation("tanh")
		self.alphas = Softmax(axis = 1, name='attention_weights')
		
		super(attention, self).build(input_shape)

	def call(self, enc_output, h_state, c_state):

		h_state_time = K.expand_dims(h_state, axis=1)
		c_state_time = K.expand_dims(c_state, axis=1)
		hc_state_time = concatenate([h_state_time, c_state_time], axis=-1)

		x1 = self.conv1d_1(enc_output)
		x2 = self.conv1d_2(hc_state_time)
		x3 = self.tanh(x1 + x2)
		x4 = self.conv1d_3(x3)
		attn_w = self.alphas(x4) 

		context = attn_w * enc_output
		context = tf.reduce_sum(context, axis=1)
		context = K.expand_dims(context, axis=1)

		return [attn_w, context]

	def compute_output_shape(self):
		return [(input_shape[0], Tx, 1), (input_shape[0], 1, n_s)]

	def get_config(self):
		config = super(attention, self).get_config()
		config.update({"hidden_units": self.hidden_units})
		return config