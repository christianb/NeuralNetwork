# Neural Network
Implementation of Tariq Rashid "Make Your Own Neural Network" using Python.

To run the benchmark execute `python Benchmark.py`

## Mnist data
To train the network you should download full train data 
[http://www.pjreddie.com/media/files/mnist_train.csv](http://www.pjreddie.com/media/files/mnist_train.csv)
and testing data [http://www.pjreddie.com/media/files/mnist_test.csv](http://www.pjreddie.com/media/files/mnist_test.csv)

Add both files to the `/data` folder or execute: `python download_mnist.py`

## Benchmark configuration
The benchmark is configurable setting:
- number of epochs
- list number of hidden nodes
- list learning rate



## Benchmark
Best Performance:  0.9783
 - learning rate:  0.05
 - epoch:  10
 - hidden nodes:  400
