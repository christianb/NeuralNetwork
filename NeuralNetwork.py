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

    def train(self, input_list, target_list):
        # convert lists to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer: w_ih x inputs
        # hidden_inputs = numpy.dot(self.w_ih, inputs)
        # calculate the signals emerging from hidden layer
        # hidden_outputs = self.activationFunction(hidden_inputs)
        hidden_outputs = self.signal_output(self.w_ih, inputs)

        # calculate signals into final output layer: w_ho x hiddenOutputs
        # final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        # calculate the signals emerging from final output layer
        # final_outputs = self.activationFunction(final_inputs)
        final_outputs = self.signal_output(self.w_ho, hidden_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined hidden nodes
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        # update the weights for the links between the hidden  and output layers
        self.w_ho += self.delta_weights(output_errors, final_outputs, hidden_outputs)
        # update the weights for the links between the input and the hidden layers
        self.w_ih += self.delta_weights(hidden_errors, hidden_outputs, inputs)

        pass

    # returns the ∂ W_jk
    # outputs_j: output from layer j
    # outputs_k: output from layer k
    def delta_weights(self, errors, outputs_k, outputs_j):
        return self.learningRate * numpy.dot(
            (errors * outputs_k * (1.0 - outputs_k)),
            numpy.transpose(outputs_j)
        )
        pass

    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer: w_ih x inputs
        # hidden_inputs = numpy.dot(self.w_ih, inputs)
        # calculate the signals emerging from hidden layer
        # hidden_outputs = self.activationFunction(hidden_inputs)

        hidden_outputs = self.signal_output(self.w_ih, inputs)

        # calculate signals into final output layer: w_ho x hiddenOutputs
        # final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        # calculate the signals emerging from final output layer
        # final_outputs = self.activationFunction(final_inputs)

        return self.signal_output(self.w_ho, hidden_outputs)
        pass

    # Calculates: w_ik * inputs
    # Applies the activation function to (w_ik * input)
    def signal_output(self, w_ik, inputs):
        signal_into_layer = numpy.dot(w_ik, inputs)  # calculate signals into layer: w_ik x inputs
        return self.activationFunction(signal_into_layer)  # calculate the signals emerging from layer
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

print(n.query([1.0, 0.5, -1.5]))
