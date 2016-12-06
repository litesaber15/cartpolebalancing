import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from adam import Adam
class NN:
	def __init__(self, dim_input, dim_hidden_layers, dim_output):
		# dim_hidden_layers in a list with ith element being no. of nodes in hidden layer i
		self.W = []
		self.B = []
		self.layers = []
		self.X = T.dmatrix()
		self.Y = T.dmatrix() # reward times action vector
		for i in range(len(dim_hidden_layers)+1):
			w = None
			lyr = None
			if i==0:
				w = theano.shared(np.array(np.random.rand(dim_input,dim_hidden_layers[0]), dtype=theano.config.floatX))
				b = theano.shared(np.zeros((dim_hidden_layers[0],), dtype=theano.config.floatX))
				lyr = self.layer(self.X, w, b)
			elif i==len(dim_hidden_layers):
				w = theano.shared(np.array(np.random.rand(dim_hidden_layers[i-1],dim_output), dtype=theano.config.floatX))
				b = theano.shared(np.zeros((dim_output,), dtype=theano.config.floatX))
				lyr = self.softmax_layer(self.layers[i-1], w, b) # output layer

			else:
				w = theano.shared(np.array(np.random.rand(dim_hidden_layers[i-1],dim_hidden_layers[i]), dtype=theano.config.floatX))
				b = theano.shared(np.zeros((dim_hidden_layers[i],), dtype=theano.config.floatX))
				lyr = self.layer(self.layers[i-1],w,b)
			self.W.append(w)
			self.B.append(b)
			self.layers.append(lyr)
		#cost equation
		loss = T.sum(T.log(T.dot(self.layers[-1],-self.Y)))#+ L1_reg*L1 + L2_reg*L2
		#loss = T.sum(T.square(self.layers[-1]-self.Y))#+ L1_reg*L1 + L2_reg*L2
		
		updates = Adam(loss, self.W) #+ Adam(loss, self.B)

		#compile theano functions
		self.backprop = theano.function(inputs=[self.X, self.Y], outputs=loss, updates=updates)
		self.run_forward = theano.function(inputs=[self.X], outputs=self.layers[-1])

	def layer(self, x, w, b):
		m = T.dot(x, w)
		m += b
		h = nnet.sigmoid(m)
		return h

	def softmax_layer(self, x, w, b):
		# last layer is softmax layer since it represents probabilities of actions to pick, and should sum to 1
		return T.nnet.softmax(self.layer(x,w,b))#.reshape((2,))

	def get_action_preds(self, state):
		#state = np.array([state])#.reshape((4,2))
		return self.run_forward(state)
dim_state = 4 #dimension of state vector
# there are two possible actions: move cart left(0) or move cart right (1)
dim_action = 2 #dimension of action vector
nn = NN(dim_input=dim_state, dim_hidden_layers=[5,3], dim_output=dim_action)
state = np.array([1,2,3,4], dtype=theano.config.floatX)
print(nn.get_action_preds(state))
