import os

os.system("curl http://www.pjreddie.com/media/files/mnist_train.csv -L -o data/mnist_train.csv")
os.system("curl http://www.pjreddie.com/media/files/mnist_test.csv -L -o data/mnist_test.csv")