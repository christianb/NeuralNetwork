# Neural Network
Implementation of Tariq Rashid "Make Your Own Neural Network" using Python.

[How does the neural network work in a nutshell](Nutshell.md "How does the neural network work in a nutshell")

## Setup
### Python Modules
You need to install the following modules:
`pip install matplotlib`
`pip install scipy`

### Mnist Data
To train the network you should download full train data 
[https://www.pjreddie.com/media/files/mnist_train.csv](https://www.pjreddie.com/media/files/mnist_train.csv)
and testing data [https://www.pjreddie.com/media/files/mnist_test.csv](https://www.pjreddie.com/media/files/mnist_test.csv)

`curl https://www.pjreddie.com/media/files/mnist_train.csv -o data/mnist_train.csv`

`curl https://www.pjreddie.com/media/files/mnist_test.csv -o data/mnist_test.csv`

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

* Visualize the weights between the layers.

* Send output signal reverse to get an image back from the input nodes.

* Make the number of hidden layers and their nodes configurable.

* Try different activation functions and see how they change the performance.

* Rotate the train images by +/- 10° to improve performance.

## Notes
* How does the number of hidden layers affect the performance?
