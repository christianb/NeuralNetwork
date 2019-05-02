# Neural Networks in a Nutshell
In short, the neural network will find the minimum error in a very complex function. 
The neural network will adjust the configuration of the network based on the error. 
With every iteration it tries to reduce the value of the error. 

## Error
The error `E` is the difference between the target value `t` and the actual value `x`.

E = t - x

t = (A + <sub>&delta;</sub>A)x

Based on the error the neural network can adjust its configuration. 
There is a relation between the error `E`, a given input value `x` and
the delta of the value <sub>&delta;</sub>A we wanna adjust.

<sub>&delta;</sub>A = E / x

## Moderation
If we just blindly apply that formula to every new input value the neural network will not learn, cause it does
not apply the previous "learned" results.
We need to **moderate** the learning with a parameter `L`.

<sub>&delta;</sub>A = L (E / x)

`L` is the learning rate. If the training data is not perfect and contains some errors the learning rate will
reduce that error. 
   
## Neurons
Biological neurons do not fire every time. They wait until a certain strenght of input has been reached. 
We need to apply the same functionality for our neural network by applying an activation function. 
One candidate as a good activiation function is the logistic sigmoid function:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\frac{1}{1+e^{-x}}" title="logistic sigmoid function" />

A neuron takes a certain number of input, accumulates the input values, applies the activation function and returns an output.

## Layers and Nodes
A neural network contains at minimum one input and one output layer.
The input layer passes the input values to the next layer.
The output layer presents the result of the neural network.
A neural network can also have hidden layers.

To represent the layers we can use a matrix. <img src="https://latex.codecogs.com/svg.latex?\Large&space;W_{ij}" title="" />

Within a matrix `W` the value represents the strength of a signal going from node `i` to node `j`. 
A value of 0 means that no signal is being emitted. 
At a beginning all matrices between the layers should be initialized. A good first approach will be a random init value.
