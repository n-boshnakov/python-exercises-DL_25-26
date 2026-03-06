import numpy as np

try:
    from . import task01, task02, task03
except ImportError:
    import task01, task02, task03

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
        for n in range(self.inputs - 1):
            w = x[n] * self.weight[n]
            total_weight += w
        return total_weight

# function created with help from Copilot
def finite_difference_derivative_mul(w, dataset, eps=1e-5):
    loss1 = calculate_loss_mul(w, dataset)

    # temporarily modify the weight
    for n in range(w.inputs):
        w.weight[n] += eps
    loss2 = calculate_loss_mul(w, dataset)

    # restore the weight
    for n in range(w.inputs):
        w.weight[n] -= eps

    return (loss2 - loss1) / eps


def calculate_loss_mul(w, dataset):
    MSE = 0
    for data in dataset:
        curr_e = (data[-1] - w.return_result(data)) ** 2
        MSE += curr_e
    return MSE / len(dataset)

def single_step(w, dataset, learning_rate=1):
    print(f"loss before: {calculate_loss_mul(w, dataset)}")
    for n in range(w.inputs):
        w.weight[n] -= learning_rate * finite_difference_derivative_mul(w, dataset)
    print(f"loss after: {calculate_loss_mul(w, dataset)}")

def train_for_epochs(w, epochs, dataset, learning_rate = 1):
    for n in range(epochs):
        single_step(w, dataset, learning_rate)
    return w

def main():
    AND_dataset = {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)}
    OR_dataset = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)}
    
    model_AND = PerceptronMul(2)

    model_OR = PerceptronMul(2)

    train_for_epochs(model_AND, 100, AND_dataset, 0.1)
    # train_for_epochs(model_OR, 100000, OR_dataset, 0.1)

    # tests
    print(f"AND for 1 and 1: {model_AND.return_result([1, 1])}")
    print(f"AND for 0 and 1: {model_AND.return_result([0, 1])}")
    print(f"AND for 1 and 0: {model_AND.return_result([1, 0])}")
    print(f"AND for 0 and 0: {model_AND.return_result([1, 1])}")
if __name__ == "__main__":
    main()