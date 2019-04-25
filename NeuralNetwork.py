import numpy

# scipy.special for the sigmoid function expit()
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inputNodes = input_nodes
        self.hiddenNodes = hidden_nodes
        self.outputNodes = output_nodes
        self.learningRate = learning_rate

        self.w_ih = self.create_weight_normal(self.inputNodes, self.hiddenNodes)
        self.w_ho = self.create_weight_normal(self.hiddenNodes, self.outputNodes)

        # activation function is the sigmoid function
        self.activationFunction = lambda x: scipy.special.expit(x)

        pass

    def train(self):
        pass

    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer: w_ih x inputs
        hidden_inputs = numpy.dot(self.w_ih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)

        # calculate signals into final output layer: w_ho x hiddenOutputs
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs
        pass

    @staticmethod
    def create_weight_simple(i, k):
        return numpy.random.rand(k, i) - 0.5
        pass

    # creates weights with from a normal distribution
    # with a standardDeviation = pow(k, -0.5)
    # and an averageValue = 0.0
    @staticmethod
    def create_weight_normal(i, k):
        average_value = 0.0
        standard_deviation = pow(k, -0.5)
        return numpy.random.normal(average_value, standard_deviation, (k, i))
        pass

    def print_weights(self):
        print("w_ih:")
        print(self.w_ih)
        print()
        print("w_ho")
        print(self.w_ho)
        pass

    pass


n = NeuralNetwork(3, 3, 3, 0.3)
n.print_weights()
