import gym
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from adam import Adam
from pprint import pprint
env = gym.make('CartPole-v0')
env.reset()
num_experiments = 1
training_batch_size = 10
num_batches = 500
e = 0.2# epsilon greedy action selection probability
e_discount = 1
discount = 1 #discount factor while calculating returns
display = False # whether to render graphic
monitor = False #whether to monitor the run and save video
#state is defined by cart position, cart velocity, pole angle, pole tip velocity
dim_state = 4 #dimension of state vector
# there are two possible actions: move cart left(0) or move cart right (1)
dim_action = 2 #dimension of action vector
#wid_hidden = 5#no. of nodes in hidden layer
L1_reg = 1 #weight of L1 regularization term
L2_reg = 1 #weight of L2 regularization term

# action_probabilities = T.dvector() # score of each action evaluated by the NN for a state
# ret = T.dscalar() # return for an action
# #ret_vector = T.dvector() # reward times action vector
# action_vector = T.dvector() # one hot vector representing which action was chosen

def layer(x, w):
	b = np.array([1], dtype=theano.config.floatX) #add bias term
	new_x = T.concatenate([x, b])
	m = T.dot(w.T, new_x)
	h = nnet.sigmoid(m)
	return h

def softmax_layer(x,w):
	# last layer is softmax layer since it represents probabilities of actions to pick, and should sum to 1
	return T.nnet.softmax(layer(x,w)).reshape((2,))

def flip(a):
	if a==0:
		return 1
	return 0

def e_greedy(action_probabilities, off=False):
	# select best action with probability 1-e
	# else select random action with probability e
	global e
	action = 1
	if action_probabilities[0] > action_probabilities[1]:
		action=0
	if off:
		return action
	from random import random
	r = random()
	if r<e:
		return flip(action)
	return action

# randomly initialize weights of hidden layer and output layer
# these are initialized as shared variables that are updated by theano
# +1 in input dimension is for bias
class NN:
	def __init__(self, dim_input, dim_hidden_layers, dim_output):
		# dim_hidden_layers in a list with ith element being no. of nodes in hidden layer i
		self.W = []
		self.layers = []
		self.X = T.dvector()
		self.Y = T.dvector() # reward times action vector
		for i in range(len(dim_hidden_layers)+1):
			w = None
			lyr = None
			if i==0:
				w = theano.shared(np.array(np.random.rand(dim_input+1,dim_hidden_layers[0]), dtype=theano.config.floatX))
				lyr = layer(self.X, w)
			elif i==len(dim_hidden_layers):
				w = theano.shared(np.array(np.random.rand(dim_hidden_layers[i-1]+1,dim_output), dtype=theano.config.floatX))
				lyr = softmax_layer(self.layers[i-1], w) # output layer

			else:
				w = theano.shared(np.array(np.random.rand(dim_hidden_layers[i-1]+1,dim_hidden_layers[i]), dtype=theano.config.floatX))
				lyr = layer(self.layers[i-1],w)
			self.W.append(w)
			self.layers.append(lyr)
		#cost equation
		loss = T.sum(T.log(T.dot(self.layers[-1],-self.Y)))#+ L1_reg*L1 + L2_reg*L2
		#loss = T.sum(T.square(self.layers[-1]-self.Y))#+ L1_reg*L1 + L2_reg*L2
		
		updates = Adam(loss, self.W)

		#compile theano functions
		self.backprop = theano.function(inputs=[self.X, self.Y], outputs=loss, updates=updates)
		self.run_forward = theano.function(inputs=[self.X], outputs=self.layers[-1])


for cv in range(num_experiments):
	nn = NN(dim_input=dim_state, dim_hidden_layers=[5,3], dim_output=dim_action)

	#train neural net
	conv = 0
	for batch_num in range(num_batches):
		e *= e_discount # randomness in action selection reduced with each batch
		max_steps = 0
		avg_steps = 0.0
		batch_ret_vectors = []
		batch_states = []
		for train_iter in range(training_batch_size):
			state = env.reset()
			rewards = []
			returns = []
			states = []
			action_vectors = []
			ret_vectors = []

			#an episode is one complete run of cart-pole 
			#a step in an episode is a point where we can take an action

			#run an episode and store states,rewards of each step
			step = 0
			done = False
			#episode begins
			while not done and step <= 200:
				states.append(state)
				if display:
					env.render()
				#get recommended action from forward pass of neural network
				action_probabilities = nn.run_forward(state)
				action = e_greedy(action_probabilities)
				action_vector = [0]*dim_action
				action_vector[action] = 1
				action_vectors.append(action_vector)
				#take the action
				state, reward, done, info = env.step(action)
				rewards.append(reward)
				step += 1
			batch_states.append(states)
			#episode ends

			avg_steps = (avg_steps*train_iter + step)/(train_iter+1)
			if step>max_steps:
				max_steps = step

		 	# calculate discounted return for each step
			for i in range(len(states)):  
				ret = 0
				future_steps = len(states) - i
				decrease = 1
				for j in xrange(future_steps):
					ret += rewards[i+j]*decrease
					decrease *= discount
					#ret_vector = np.dot(ret, action_vectors[i])
					#ret_vectors.append(ret_vector)
					ret_vectors.append(action_vectors[i])
			batch_ret_vectors.append(ret_vectors)

		#backprop discounted return
		for i in range(training_batch_size): #for each episode in batch
			for j in range(len(batch_states[i])): # for each step of episode
				nn.backprop(batch_states[i][j],batch_ret_vectors[i][j])

		#print('Batch: '+str(batch_num)+' Max: '+str(max_steps)+ ' Avg: '+str(avg_steps)+' Epsilon: '+str(e))
		if conv>15:
			break
		elif avg_steps < 180:
			conv = 0
		else:
			conv +=1

	# test learnt policy, e-greedy is off
	#print('********\nTesting\n********')
	if monitor:
		env.monitor.start('/home/kartikeya/keras/cartpole-monitor', force=True)
	max_steps = 0
	avg_steps = 0.0
	step=0
	for test_iter in range(100):		
		state = env.reset()
		done = False
		import time
		step = 0
		while not done and step<200:
			step +=1 
			if test_iter == 99:
				pass
				#time.sleep(0.001)
				#env.render()
			action_probabilities = nn.run_forward(state)
			action = e_greedy(action_probabilities, off=True)
			state, reward, done, info = env.step(action)
		avg_steps = (avg_steps*test_iter + step)/(test_iter+1.0)
		if step>max_steps:
			max_steps = step
		if test_iter == 99:
			print('Steps: '+str(step)+ ' Max: '+str(max_steps)+ ' Avg: '+str(avg_steps))
	if monitor:
		env.monitor.close()