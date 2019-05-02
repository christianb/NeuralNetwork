import matplotlib.pyplot
import numpy
import FileReader
from NeuralNetwork import NeuralNetwork


class Mnist:
    def __init__(self, hidden_nodes, learning_rate):
        input_nodes = 784  # input data has 28 * 28 pixel
        self.output_nodes = 10  # output data are the numbers from 0..9

        self.train_data_list = Mnist.get_train_data_list()
        self.test_data_list = Mnist.get_test_data_list()
        self.neural_network = NeuralNetwork(input_nodes, hidden_nodes, self.output_nodes, learning_rate)

    def train(self):
        for record in self.train_data_list:
            all_values = record.split(',')  # split the record by the ',' commas

            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift he inputs

            targets = numpy.zeros(self.output_nodes) + 0.01

            targets[int(all_values[0])] = 0.99

            self.neural_network.train(inputs, targets)

    def test(self):
        scorecard = []

        for record in self.test_data_list:

            all_values = record.split(',')  # split the record by the ',' commas

            correct_label = int(all_values[0])  # correct answer is the first label

            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # query the network
            outputs = self.neural_network.query(inputs)

            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)

            # append correct or incorrect answer to list
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)

        # calculates the performance score, the fraction of correct answers
        scorecard_array = numpy.asarray(scorecard)

        # return performance
        return scorecard_array.sum() / scorecard_array.size

    # run the network backwards, given a label, see what image it produces
    def backward(self, label):
        # create the output signals for this label
        targets = numpy.zeros(self.output_nodes) + 0.01

        # all_values[0] is the target label for this record
        targets[label] = 0.99

        # get image data
        image_data = self.neural_network.backquery(targets)

        # plot image data
        matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
        matplotlib.pyplot.show()

    @staticmethod
    def get_train_data_list():
        optional_train_file_name = "data/mnist_train.csv"
        default_train_file_name = "data/mnist_train_100.csv"
        return FileReader.read_optional_file_or_default(optional_train_file_name, default_train_file_name)

    @staticmethod
    def get_test_data_list():
        optional_test_file_name = "data/mnist_test.csv"
        default_test_file_name = "data/mnist_test_10.csv"
        return FileReader.read_optional_file_or_default(optional_test_file_name, default_test_file_name)
