import FileReader
from NeuralNetwork import NeuralNetwork


class Mnist:
    def __init__(self):
        self.input_nodes = 784  # input data has 28 * 28 pixel
        self.output_nodes = 10  # output data are the numbers from 0..9

    pass

    def create_neural_network(self, hidden_nodes, learning_rate):
        return NeuralNetwork(self.input_nodes, hidden_nodes, self.output_nodes, learning_rate)

    pass

    @staticmethod
    def get_train_data_list():
        optional_train_file_name = "data/mnist_train.csv"
        default_train_file_name = "data/mnist_train_100.csv"
        return FileReader.read_optional_file_or_default(optional_train_file_name, default_train_file_name)

    pass

    @staticmethod
    def get_test_data_list():
        optional_test_file_name = "data/mnist_test.csv"
        default_test_file_name = "data/mnist_test_10.csv"
        return FileReader.read_optional_file_or_default(optional_test_file_name, default_test_file_name)

    pass


pass
