import gym
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
e = 0.1 # epsilon greedy action selection probability
display = False # whether to render graphic
dim_state = 4
dim_action = 2
wid_hidden = 4

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
	alpha = 0.2 #learning rate
	return theta - (alpha * T.grad(cost, wrt=theta))

def flip(a):
	if a==0:
		return 1
	return 0

def e_greedy(action_probabilities):
	global e
	action = 1
	if action_probabilities[0] > action_probabilities[1]:
		action=0
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
mse_fc = -T.sum(fc)

#compile theano functions
cost = theano.function(inputs=[state,ret], outputs=mse_fc, 
	updates=[(w1, grad_desc(mse_fc, w1)), (w2, grad_desc(mse_fc, w2))])

run_forward = theano.function(inputs=[state], outputs=output_layer)

max_steps = 0
for train_iter in range(20000):
	state = env.reset()
	actions = []
	rewards = []
	returns = []
	states = []
	a_probs = []
	total_reward = 0

	#episode = 1 complete run of cart-pole policy
	#epoch = each step

	#run an episode and store action taken at each step
	#along with its corresponding reward

	step = 0
	done = False
	while not done and step <= 200:
		if display:
			env.render()
		# select best action with probability 1-e
		# random action with probability e
		action_probabilities = run_forward(state)
		states.append(state)
		action = e_greedy(action_probabilities)
		action_vector = np.zeros(dim_action)
		action_vector[action] = 1
		action_probabilities = action_vector*action_probabilities
		#actions.append(action)
		a_probs.append(action_probabilities)
		state, reward, done, info = env.step(action)
		rewards.append(reward)
		total_reward += reward
		step += 1
	if step>max_steps:
		max_steps = step
	print('Iter: '+str(train_iter)+' Steps: '+str(step)+' Max steps: '+str(max_steps))

 	# calculate discounted return for each step
	for i in range(len(states)):  
		ret = 0
		future_steps = len(actions) - i
		decrease = 1
		for j in xrange(future_steps):
			ret += rewards[i+j]*decrease
			decrease *= 0.97

		#backprop this discounted return
		curr_cost = cost(states[i],ret)
		#curr_cost = cost(a_probs[i], ret)
		#returns.append(ret)

# balance pole with learned policy
state = env.reset()
done = False
import time
steps = 0
while not done:
	steps +=1 
	time.sleep(0.001)
	env.render()
	action = 1
	action_probabilities = run_forward(state)
	if action_probabilities[0] > action_probabilities[1]:
		action = 0
	state, reward, done, info = env.step(action)
print(steps)

