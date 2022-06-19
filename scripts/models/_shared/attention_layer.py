import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation, concatenate, Dot



class attention(tf.keras.layers.Layer):

	def __init__(self, hidden_units, **kwargs):
		# super(attention, self).__init__(hidden_units)
		self.hidden_units = hidden_units
		super(attention, self).__init__(**kwargs)


	def build(self, input_shape):

		input_dim = int(input_shape[-1])

		self.attention_score_vec = Dense(64, name='attention_score_vec')
		self.h_t = Dense(64, name='ht')
		self.attention_score = Dot(axes=[1, 2], name='attention_score')
		self.attention_weight = Activation('softmax', name='attention_weight')
		self.context_vector = Dot(axes=[1, 1], name='context_vector')
		self.attention_vector = Dense(self.hidden_units, activation='tanh', name='attention_vector')

		super(attention, self).build(input_shape)

	def call(self, enc_output, enc_out, h_state, c_state):


		score_first_part = self.attention_score_vec(enc_output)
        # score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
		h_t = concatenate([h_state, enc_out[:,0,:]])
		h_t = self.h_t(h_t)

		score = self.attention_score([h_t, score_first_part])

		attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
		context_vector = self.context_vector([enc_output, attention_weights])
		pre_activation = concatenate([context_vector, h_t])
		attention_vector = self.attention_vector(pre_activation)

		attention_weights = K.expand_dims(attention_weights, axis=-1)
		attention_vector = K.expand_dims(attention_vector, axis=1)

		return [attention_weights, attention_vector]

	def compute_output_shape(self):
		return [(input_shape[0], Tx, 1), (input_shape[0], 1, n_s)]

	def get_config(self):
		config = super(attention, self).get_config()
		config.update({"hidden_units": self.hidden_units})
		return config
