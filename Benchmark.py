from Mnist import Mnist
import numpy

# Benchmark configuration

epochs = 10
hidden_nodes_list = [100, 200, 400]  # default 100
learning_rate_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
train_data_list = Mnist.get_train_data_list()
test_data_list = Mnist.get_test_data_list()

mnist = Mnist()

best_performance = 0
best_learning_rate = 0
best_hidden_nodes = 0
best_nr_of_epochs = 0

for h_nodes in hidden_nodes_list:
    print("hidden nodes: ", h_nodes)

    for lr in learning_rate_list:
        print("  learning rate: ", lr)
        n = mnist.create_neural_network(h_nodes, lr)

        performance_list = []

        for e in range(epochs):
            print("    run training epoch: ", e + 1)

            # train the neural network

            for record in train_data_list:
                all_values = record.split(',')  # split the record by the ',' commas

                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift he inputs

                targets = numpy.zeros(mnist.output_nodes) + 0.01

                targets[int(all_values[0])] = 0.99

                n.train(inputs, targets)
                pass

            # test the neural network

            print("    start testing...")

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

            # calculates the performance score, the fraction of correct answers
            scorecard_array = numpy.asarray(scorecard)
            performance = scorecard_array.sum() / scorecard_array.size

            if performance > best_performance:
                best_performance = performance
                best_hidden_nodes = h_nodes
                best_nr_of_epochs = e
                best_learning_rate = lr

                pass

            performance_list.append(performance)
            print("    performance = ", performance)
            print()

        pass
    pass

pass

print("Best Performance: ", best_performance)
print(" with learning rate: ", best_learning_rate)
print(" with epoche: ", best_nr_of_epochs + 1, " (", epochs, ")")
print(" with hidden nodes: ", best_hidden_nodes)
