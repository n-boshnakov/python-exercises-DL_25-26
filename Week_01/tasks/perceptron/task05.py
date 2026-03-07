import numpy as np

def initialize_weights(x, y):
    return np.random.uniform(x, y)

class PerceptronMul():
    def __init__(self, inputs):
        self.inputs = inputs
        self.weight = []
        for n in range(self.inputs):
            self.weight.append(initialize_weights(0, 10))

    def return_result(self, x):
        total_weight = 0
        for n in range(self.inputs):
            w = x[n] * self.weight[n]
            total_weight += w
        return total_weight

def finite_difference_derivative_mul(w, dataset, eps=1e-5):
    gradients = []

    for n in range(w.inputs):
        original_weight = w.weight[n]

        loss1 = calculate_loss_mul(w, dataset)

        w.weight[n] = original_weight + eps
        loss2 = calculate_loss_mul(w, dataset)

        w.weight[n] = original_weight

        grad_i = (loss2 - loss1) / eps
        gradients.append(grad_i)

    return gradients

def calculate_loss_mul(w, dataset):
    MSE = 0
    for data in dataset:
        curr_e = (data[-1] - w.return_result(data)) ** 2
        MSE += curr_e
    return MSE / len(dataset)

def single_step(w, dataset, learning_rate=1):
    print(f"loss before: {calculate_loss_mul(w, dataset)}")
    grads = finite_difference_derivative_mul(w, dataset)

    for i in range(w.inputs):
        w.weight[i] -= learning_rate * grads[i]

    print(f"loss after: {calculate_loss_mul(w, dataset)}")

def train_for_epochs(w, epochs, dataset, learning_rate = 1):
    for n in range(epochs):
        single_step(w, dataset, learning_rate)
    return w

def main():
    AND_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    OR_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    
    # each perceptron has 2 parameters
    model_AND = PerceptronMul(2)

    model_OR = PerceptronMul(2)

    train_for_epochs(model_AND, 100000, AND_dataset, 0.01)
    train_for_epochs(model_OR, 100000, OR_dataset, 0.01)

    # tests
    print(f"AND for 1 and 1: {model_AND.return_result([1, 1])}")
    print(f"AND for 0 and 1: {model_AND.return_result([0, 1])}")
    print(f"AND for 1 and 0: {model_AND.return_result([1, 0])}")
    print(f"AND for 0 and 0: {model_AND.return_result([0, 0])}")

    print(f"OR for 1 and 1: {model_OR.return_result([1, 1])}")
    print(f"OR for 0 and 1: {model_OR.return_result([0, 1])}")
    print(f"OR for 1 and 0: {model_OR.return_result([1, 0])}")
    print(f"OR for 0 and 0: {model_OR.return_result([0, 0])}")

    # the values the models return are around 0 and 1, close to the answers, but do not predict them exactly
if __name__ == "__main__":
    main()