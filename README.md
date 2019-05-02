# Neural Network
Implementation of Tariq Rashid "Make Your Own Neural Network" using Python.

[How does the neural network work in a nutshell](Nutshell.md "How does the neural network work in a nutshell")

## Mnist data
To train the network you should download full train data 
[http://www.pjreddie.com/media/files/mnist_train.csv](http://www.pjreddie.com/media/files/mnist_train.csv)
and testing data [http://www.pjreddie.com/media/files/mnist_test.csv](http://www.pjreddie.com/media/files/mnist_test.csv)

Add both files to the `/data` folder or execute: `python download_mnist.py`

## Benchmark 
To run the benchmark execute `python Benchmark.py`

You can configure the benchmark by changing the following constants:
- number of `epochs`
- `hidden_nodes_list`
- `learning_rate_list`

Best performance:  0.9783
 - epoch:  10
 - hidden_nodes:  400
 - learning_rate:  0.05

## Tasks
* Destroy single or a specific amount of nodes/weights to see how much it affects performance.

* Visualize the weights between the layers

* Send output signal reverse to get an image back from the input nodes

* Make the number of hidden layers and their nodes configurable.

* Try different activation functions and see how they change the performance.

## Notes
* How does the number of hidden layers affect the performance?