# cartpolebalancing
Using Policy Iteration to solve the Cart Pole Balancing problem. A simple 1 hidden layer fully connected neural network is used to evaluate the best action for a given state. Suppose a training episode lasts for `k` steps. Reward for each step is collected, and discounted return is calculated for each step after the episode ends. (state,discounted return) is stored for each each episode. Backpropogration is done for a batch of episodes, and the process is repeated for a number of batches. 

Here's a GIF of the trained AI:
![Screen GIF](/screen.gif?raw=true "GIF of trained AI")

Simulation environment: [OpenAI Gym Cartpole-v0](https://gym.openai.com/envs/CartPole-v0)

Forward pass and backpropogation done in Theano. Here are good tutorial for [getting started with Theano](http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb) and for [implementing a simple ANN](http://outlace.com/Beginner-Tutorial-Theano/).

I used the CPU for this. The Nvidia drivers are a bit tricky to install on Ubuntu 1604 if you have Intel's Skylake. Here's my Theano `.theanorc` config for CPU:
```
[global]
floatX = float32
device = cpu
force_device=True
pycuda.init = False

[lib]
cnmem = 1

[blas]
ldflags=-L/usr/lib/ -lblas
```
