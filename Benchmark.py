from Mnist import Mnist
import time

# Benchmark configuration

epochs = 10
hidden_nodes_list = [400]  # default 100
learning_rate_list = [0.05]

best_performance = 0
best_learning_rate = 0
best_hidden_nodes = 0
best_nr_of_epochs = 0

timestamp = time.time()

for h_nodes in hidden_nodes_list:
    print("hidden nodes: ", h_nodes)

    for lr in learning_rate_list:
        print("  learning rate: ", lr)
        mnist = Mnist(h_nodes, lr)

        performance_list = []

        for e in range(epochs):
            print("    train epoch: ", e + 1)

            # train the neural network
            train_start_time = time.time()
            mnist.train()
            print("    training took:", time.time() - train_start_time, "ms")

            # test the neural network
            test_start_time = time.time()
            performance = mnist.test()
            print("    test took:", time.time() - test_start_time, "ms")

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
print(" with epoch: ", best_nr_of_epochs + 1, " (", epochs, ")")
print(" with hidden nodes: ", best_hidden_nodes)


print("time to run:", time.time() - timestamp, "ms")
