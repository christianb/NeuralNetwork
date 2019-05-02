import numpy

# scipy.special for the sigmoid function expit()
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.learningRate = learning_rate

        self.w_ih = self.create_weight_normal(input_nodes, hidden_nodes)
        self.w_ho = self.create_weight_normal(hidden_nodes, output_nodes)

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # the inverse sigmoid function logit
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    def train(self, input_list, target_list):
        # convert lists to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer: w_ih x inputs
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.signal_output(self.w_ih, inputs)

        # calculate signals into final output layer: w_ho x hiddenOutputs
        # calculate the signals emerging from final output layer
        final_outputs = self.signal_output(self.w_ho, hidden_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined hidden nodes
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        # update the weights for the links between the hidden  and output layers
        self.w_ho += self.delta_weights(output_errors, final_outputs, hidden_outputs)
        # update the weights for the links between the input and the hidden layers
        self.w_ih += self.delta_weights(hidden_errors, hidden_outputs, inputs)

    # returns the ∂ W_jk
    # outputs_j: output from layer j
    # outputs_k: output from layer k
    def delta_weights(self, errors, outputs_k, outputs_j):
        return self.learningRate * numpy.dot(
            (errors * outputs_k * (1.0 - outputs_k)),
            numpy.transpose(outputs_j)
        )

    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer: w_ih x inputs
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.signal_output(self.w_ih, inputs)

        # calculate signals into final output layer: w_ho x hiddenOutputs
        # calculate the signals emerging from final output layer
        return self.signal_output(self.w_ho, hidden_outputs)

    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.w_ho.T, final_inputs)

        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.w_ih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    # Calculates: w_ik * inputs
    # Applies the activation function to (w_ik * input)
    def signal_output(self, w_ik, inputs):
        signal_into_layer = numpy.dot(w_ik, inputs)  # calculate signals into layer: w_ik x inputs
        return self.activation_function(signal_into_layer)  # calculate the signals emerging from layer

    # Applies the inverse activation function
    def signal_input(self, w_ik, outputs):
        inputs = self.inverse_activation_function(outputs)
        return numpy.dot(w_ik.T, inputs)

    def save_to_file(self):
        numpy.save("data/saved_w_ih.npy", self.w_ih)
        numpy.save("data/saved_w_ho.npy", self.w_ho)

    def load_from_file(self):
        self.w_ih = numpy.load("data/saved_w_ih.npy")
        self.w_ho = numpy.load("data/saved_w_ho.npy")

    @staticmethod
    def create_weight_simple(i, k):
        return numpy.random.rand(k, i) - 0.5

    # creates weights with from a normal distribution
    # with a standardDeviation = pow(k, -0.5)
    # and an averageValue = 0.0
    @staticmethod
    def create_weight_normal(i, k):
        average_value = 0.0
        standard_deviation = pow(k, -0.5)
        return numpy.random.normal(average_value, standard_deviation, (k, i))

    def print_weights(self):
        print("w_ih:")
        print(self.w_ih)
        print()
        print("w_ho")
        print(self.w_ho)
