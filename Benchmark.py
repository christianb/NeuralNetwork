from Mnist import Mnist

# Benchmark configuration

epochs = 1
hidden_nodes_list = [100]  # default 100
learning_rate_list = [0.3]

best_performance = 0
best_learning_rate = 0
best_hidden_nodes = 0
best_nr_of_epochs = 0

for h_nodes in hidden_nodes_list:
    print("hidden nodes: ", h_nodes)

    for lr in learning_rate_list:
        print("  learning rate: ", lr)
        mnist = Mnist(h_nodes, lr)

        performance_list = []

        for e in range(epochs):
            print("    run training epoch: ", e + 1)

            # train the neural network
            mnist.train()

            # test the neural network
            print("    start testing...")
            performance = mnist.test()

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
