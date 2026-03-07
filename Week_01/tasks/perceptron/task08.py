import numpy as np
import matplotlib.pyplot as plt

try:
    from . import task06, task05
except ImportError:
    import task06, task05

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class PerceptronMulSigmoid(task06.PerceptronMul):
    def return_result(self, x):
        linear_result = super().return_result(x)

        return sigmoid(linear_result)

def calculate_loss_mul_sigmoid(w, dataset):
    MSE = 0
    for data in dataset:
        curr_e = (data[-1] - w.return_result(data)) ** 2
        MSE += curr_e
    return MSE / len(dataset)

def finite_difference_derivative_mul_sigmoid(w, dataset, eps=1e-5):
    gradients = []

    for n in range(w.inputs):
        original_weight = w.weight[n]

        loss1 = calculate_loss_mul_sigmoid(w, dataset)

        w.weight[n] = original_weight + eps
        loss2 = calculate_loss_mul_sigmoid(w, dataset)

        w.weight[n] = original_weight

        grad_i = (loss2 - loss1) / eps
        gradients.append(grad_i)

    original_bias = w.bias

    loss1 = calculate_loss_mul_sigmoid(w, dataset)
    w.bias = original_bias + eps
    loss2 = calculate_loss_mul_sigmoid(w, dataset)
    w.bias = original_bias

    gradients.append((loss2 - loss1) / eps)

    return gradients

def train_for_epochs_loss_record(w, epochs, dataset,  fun, learning_rate = 1):
    loss_history = []
    for n in range(epochs):
        loss_history.append(float(fun(w, dataset, learning_rate)))
    return w, loss_history

def single_step_sigmoid(w, dataset, learning_rate=1):
    print(f"loss before: {calculate_loss_mul_sigmoid(w, dataset)}")

    grads = finite_difference_derivative_mul_sigmoid(w, dataset)

    for n in range(w.inputs):
        w.weight[n] -= learning_rate * grads[n]

    # update bias
    w.bias -= learning_rate * grads[-1]

    print(f"loss after: {calculate_loss_mul_sigmoid(w, dataset)}")
    return calculate_loss_mul_sigmoid(w, dataset)

def plot(x, y):
    plt.plot(x, y)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000], ["10k", "20k", "30k", "40k", "50k", "60k", "70k", "80k", "90k", "100k"])
    plt.title(f"Loss over epochs")
    plt.show()

def main():
    AND_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    OR_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]

    epochs = np.arange(start=0, stop=100000, step=1, dtype=None)

    model_AND_sigmoid = PerceptronMulSigmoid(2)
    model_OR_sigmoid = PerceptronMulSigmoid(2)

    loss_hist_AND = train_for_epochs_loss_record(model_AND_sigmoid, 100000, AND_dataset, single_step_sigmoid, 0.1)[1]
    loss_hist_OR = train_for_epochs_loss_record(model_OR_sigmoid, 100000, OR_dataset, single_step_sigmoid, 0.1)[1]
    print(loss_hist_AND)
    # tests
    print(f"AND for 1 and 1: {model_AND_sigmoid.return_result([1, 1])}")
    print(f"AND for 0 and 1: {model_AND_sigmoid.return_result([0, 1])}")
    print(f"AND for 1 and 0: {model_AND_sigmoid.return_result([1, 0])}")
    print(f"AND for 0 and 0: {model_AND_sigmoid.return_result([0, 0])}")
    plot(epochs, loss_hist_AND)

    print(f"OR for 1 and 1: {model_OR_sigmoid.return_result([1, 1])}")
    print(f"OR for 0 and 1: {model_OR_sigmoid.return_result([0, 1])}")
    print(f"OR for 1 and 0: {model_OR_sigmoid.return_result([1, 0])}")
    print(f"OR for 0 and 0: {model_OR_sigmoid.return_result([0, 0])}")
    plot(epochs, loss_hist_OR)

if __name__ == "__main__":
    main()