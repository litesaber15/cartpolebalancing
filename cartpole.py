import gym
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from adam import Adam
from pprint import pprint
env = gym.make('CartPole-v0')
env.reset()

training_batch_size = 10
num_batches = 100
e = 0.3# epsilon greedy action selection probability
e_discount = 1
discount = 1 #discount factor while calculating returns
display = False # whether to render graphic
monitor = True #whether to monitor the run and save video
#state is defined by cart position, cart velocity, pole angle, pole tip velocity
dim_state = 4 #dimension of state vector
# there are two possible actions: move cart left(0) or move cart right (1)
dim_action = 2 #dimension of action vector
wid_hidden = 5#no. of nodes in hidden layer
L1_reg = 1 #weight of L1 regularization term
L2_reg = 1 #weight of L2 regularization term

state = T.dvector()
action_probabilities = T.dvector() # score of each action evaluated by the NN for a state
ret = T.dscalar() # return for an action
ret_vector = T.dvector() # reward times action vector
action_vector = T.dvector() # one hot vector representing which action was chosen

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
w1 = theano.shared(np.array(np.random.rand(dim_state+1,wid_hidden), dtype=theano.config.floatX))
w2 = theano.shared(np.array(np.random.rand(wid_hidden+1,dim_action), dtype=theano.config.floatX))

hidden_layer = layer(state, w1)
output_layer = softmax_layer(hidden_layer, w2)


L1 = abs(w1).sum() + abs(w2).sum() #L1 norm regularization term
L2 = (w1**2).sum() + (w2**2).sum() #L2 norm regularization term

#cost equation
mse_fc = T.sum(T.log(T.dot(output_layer,-ret_vector)))#+ L1_reg*L1 + L2_reg*L2
updates = Adam(mse_fc, [w1,w2])

#compile theano functions
backprop = theano.function(inputs=[state,ret_vector], outputs=mse_fc, updates=updates)
run_forward = theano.function(inputs=[state], outputs=output_layer)

#train neural net
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
			action_probabilities = run_forward(state)
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
				ret_vector = np.dot(ret, action_vectors[i])
				ret_vectors.append(ret_vector)
		batch_ret_vectors.append(ret_vectors)

	#backprop discounted return
	for i in range(training_batch_size): #for each episode in batch
		for j in range(len(batch_states[i])): # for each step of episode
			backprop(batch_states[i][j],batch_ret_vectors[i][j])

	print('Batch: '+str(batch_num)+' Max: '+str(max_steps)+ ' Avg: '+str(avg_steps)+' Epsilon: '+str(e))

# test learnt policy, e-greedy is off
print('********\nTesting\n********')
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
			time.sleep(0.001)
			env.render()
		action_probabilities = run_forward(state)
		action = e_greedy(action_probabilities, off=True)
		state, reward, done, info = env.step(action)
	avg_steps = (avg_steps*test_iter + step)/(test_iter+1.0)
	if step>max_steps:
		max_steps = step
	print('Steps: '+str(step)+ ' Max: '+str(max_steps)+ ' Avg: '+str(avg_steps))
if monitor:
	env.monitor.close()