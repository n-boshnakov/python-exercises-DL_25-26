import numpy as np

# task 04 -> commented out
# np.random.seed(42)

# task 01

# The model has 1 parameters -> the single weight initialized by initialize_weights(x, y)

def create_dataset(n):
    results_list = []
    for n in range(n):
        a = n
        b = n * 2
        results_list.append((a, b))
    return results_list

def initialize_weights(x, y):
    return 	np.random.uniform(x, y)

# task 02

def calculate_loss(w, dataset):
    MSE = 0
    for data in dataset:
        curr_e = (data[1] - w.return_result(data[0])) ** 2
        MSE += curr_e
    return MSE / len(dataset)



class Perceptron():
    def __init__(self):
        self.weight = initialize_weights(0, 10)
    
    def return_result(self, x):
        return x * self.weight
    
# task 03
# function created with help from Copilot
def finite_difference_derivative(w, dataset, eps=1e-5):
    loss1 = calculate_loss(w, dataset)
    
    # temporarily modify the weight
    w.weight += eps
    loss2 = calculate_loss(w, dataset)
    
    # restore the weight
    w.weight -= eps
    
    return (loss2 - loss1) / eps

def single_step(w, dataset, learning_rate=1):
    print(f"loss before: {calculate_loss(w, dataset)}")
    w.weight -= learning_rate * finite_difference_derivative(w, dataset)
    print(f"loss after: {calculate_loss(w, dataset)}")

def train_for_epochs(w, epochs, dataset, learning_rate = 1):
    for n in range(epochs):
        single_step(w, dataset, learning_rate)
    return w

if __name__ == '__main__':
    dataset = create_dataset(6)
    w = Perceptron()

    loss = calculate_loss(w, dataset)

    print(f'MSE: {loss}')

    # task 03; testing without a learning_rate (equal to 1)
    train_for_epochs(w, 10, dataset)

    # implementing a test for adding a learning rate -> learning slows down drastically
    w1 = Perceptron()
    print("Test for learning_rate = 0.001")
    train_for_epochs(w1, 10, dataset, 0.001)

    print(f"After training for 10 epochs with learning_rate = 0.001: 5*2 = {w1.return_result(5)}")

    # implementing a test for adding a bigger learning rate -> learning is much better
    w2 = Perceptron()
    print("Test for learning_rate = 0.01")
    train_for_epochs(w2, 10, dataset, 0.01)
    print(f"After training for 10 epochs with learning_rate = 0.01: 5*2 = {w2.return_result(5)}")

    # task 04
    train_for_epochs(w, 500, dataset, 0.01)

    # task 05

    model1_AND = Perceptron()
    model2_AND = Perceptron()

    model1_OR = Perceptron()
    model2_OR = Perceptron()

    AND_dataset = {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)}
    OR_dataset = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)}
