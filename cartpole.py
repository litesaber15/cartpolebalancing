import gym
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from adam import adam

env = gym.make('CartPole-v0')
env.reset()
training_steps = 4000
e = 0.3 # epsilon greedy action selection probability
discount = 0.97 #discount factor while calculating returns
alpha = 0.3 #learning rate for grad descent in backprop
display = False # whether to render graphic
#state is defined by cart position, cart velocity, pole angle, pole tip velocity
dim_state = 4 #dimension of state vector
# there are two possible actions: move cart left(0) or move cart right (1)
dim_action = 2 #dimension of action vector
wid_hidden = 3 	#no. of nodes in hidden layer

state = T.dvector()
action_probabilities = T.dvector() # predicted value
target_actions = T.dvector()
ret = T.dscalar()

def layer(x, w):
	b = np.array([1], dtype=theano.config.floatX)
	new_x = T.concatenate([x, b])
	m = T.dot(w.T, new_x)
	h = nnet.sigmoid(m)
	return h

def softmax_layer(x,w):
	return T.nnet.softmax(layer(x,w)).reshape((2,))

def grad_desc(cost, theta):
	global alpha #learning rate
	return theta - (alpha * T.grad(cost, wrt=theta))

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

# randomly initialize weights of two hidden layers
# these are initialized as shared variables that are updated by theano
# +1 in input dimension is for bias
w1 = theano.shared(np.array(np.random.rand(dim_state+1,wid_hidden), dtype=theano.config.floatX))
w2 = theano.shared(np.array(np.random.rand(wid_hidden+1,dim_action), dtype=theano.config.floatX))

hidden_layer = layer(state, w1)
output_layer = softmax_layer(hidden_layer, w2)

#cost equation
fc = output_layer*ret #does this seem correct?
#fc = action_probabilities*ret
#mse_fc = -T.sum(T.dot(fc.T,fc))
#mse_fc = -T.sum(fc)

# todo: include l1 and l2 norm regularization terms here
mse_fc = T.sum(T.square(output_layer-ret))

#compile theano functions
#updates = [(w1,adam(mse_fc, [w1])[0]),(w2,adam(mse_fc, [w2])[0])]
updates=[(w1, grad_desc(mse_fc, w1)), (w2, grad_desc(mse_fc, w2))]
backprop = theano.function(inputs=[state,ret], outputs=mse_fc, updates=updates)
run_forward = theano.function(inputs=[state], outputs=output_layer)

max_steps = 0
avg_steps = 0
for train_iter in range(training_steps):
	state = env.reset()
	rewards = []
	states = []

	#an episode is one complete run of cart-pole 
	#epoch = each step in an episode where we take one action

	#run an episode and store states,rewards of each step
	step = 0
	done = False
	while not done and step <= 200:
		states.append(state)
		if display:
			env.render()
		#get recommended action from forward pass of neural network
		action_probabilities = run_forward(state)
		action = e_greedy(action_probabilities)
		#take the action
		state, reward, done, info = env.step(action)
		rewards.append(reward)
		step += 1

	#compute and print some metrics
	avg_steps = (avg_steps*train_iter + step)/(train_iter+1)
	if step>max_steps:
		max_steps = step
	print('Iter: '+str(train_iter)+' Steps: '+str(step)+' Max: '+str(max_steps)+ ' Avg: '+str(avg_steps))

 	# calculate discounted return for each step
	for i in range(len(states)):  
		ret = 0
		future_steps = len(states) - i
		decrease = 1
		for j in xrange(future_steps):
			ret += rewards[i+j]*decrease
			decrease *= discount

		#backprop this discounted return
		backprop(states[i],ret)
		"""
		backprop currently done epoch by epoch
		hacky, and not optimized for speed
		okay for now since cpu is being used
		"""

# test learnt policy, e-greedy is off
max_steps = 0
avg_steps = 0
for test_iter in range(500):
	state = env.reset()
	done = False
	import time
	steps = 0
	while not done:
		steps +=1 
		if test_iter == 199:
			time.sleep(0.001)
			env.render()
		action_probabilities = run_forward(state)
		action = e_greedy(action_probabilities)
		state, reward, done, info = env.step(action)
		avg_steps = (avg_steps*test_iter + step)/(test_iter+1)
		if step>max_steps:
			max_steps = step
print('Max: '+str(max_steps)+ ' Avg: '+str(avg_steps))