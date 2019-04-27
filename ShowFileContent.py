import numpy
import matplotlib.pyplot

data_file = open("data/mnist_train_100.csv", 'r')
data_list = data_file.readlines()  # do not read whole files in memory!
data_file.close()

# print(len(data_list))
print(data_list[0])

all_values = data_list[27].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.title('Zeichen')
matplotlib.pyplot.show()