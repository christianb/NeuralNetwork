import FileReader
from NeuralNetwork import NeuralNetwork
import numpy
import matplotlib.pyplot as plt

# Constants

input_nodes = 784
hidden_nodes = 100  # default: 100
output_nodes = 10

epochs = 1
learning_rate_list = [0.2, 0.1]

plot_file_name = "plot.png"

optional_train_file_name = "data/mnist_train.csv"
default_train_file_name = "data/mnist_train_100.csv"
train_data_list = FileReader.read_optional_file_or_default(optional_train_file_name, default_train_file_name)

optional_test_file_name = "data/mnist_test.csv"
default_test_file_name = "data/mnist_test_10.csv"
test_data_list = FileReader.read_optional_file_or_default(optional_test_file_name, default_test_file_name)

# train the neural network

for lr in learning_rate_list:
    print("learning rate: ", lr)
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)

    performance_list = []

    for e in range(epochs):
        print(" run training epoch: ", e + 1)

        for record in train_data_list:
            all_values = record.split(',')  # split the record by the ',' commas

            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift he inputs

            targets = numpy.zeros(output_nodes) + 0.01

            targets[int(all_values[0])] = 0.99

            n.train(inputs, targets)
            pass

        # test the neural network

        print(" start testing...")

        scorecard = []

        # go through all the records in the test data set
        for record in test_data_list:

            all_values = record.split(',')  # split the record by the ',' commas

            correct_label = int(all_values[0])  # correct answer is the first label
            # print(correct_label, " is correct value")

            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # query the network
            outputs = n.query(inputs)

            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # print(label, " is networks answer")

            # append correct or incorrect answer to list
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
                pass

            pass

        # print(scorecard)

        # calculates the performance score, the fraction of correct answers
        scorecard_array = numpy.asarray(scorecard)
        performance = scorecard_array.sum() / scorecard_array.size
        performance_list.append(performance)
        print(" performance = ", performance)
        print()

    pass

    epoch_list = list(range(1, epochs + 1))
    plt.plot(epoch_list, performance_list, label=lr)
pass

plt.ylabel("performance")
plt.xlabel("epoch")
plt.legend()
plt.savefig(plot_file_name)
