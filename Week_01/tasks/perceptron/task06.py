try:
    from . import task05
except ImportError:
    import task05

class PerceptronMul():
    def __init__(self, inputs):
        self.inputs = inputs
        self.weight = []
        self.bias = task05.initialize_weights(0, 10)
        for n in range(self.inputs):
            self.weight.append(task05.initialize_weights(0, 10))

    def return_result(self, x):
        total_weight = 0
        for n in range(self.inputs):
            w = x[n] * self.weight[n]
            total_weight += w
        return total_weight + self.bias


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

    original_bias = w.bias

    loss1 = calculate_loss_mul(w, dataset)
    w.bias = original_bias + eps
    loss2 = calculate_loss_mul(w, dataset)
    w.bias = original_bias

    gradients.append((loss2 - loss1) / eps)

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

    for n in range(w.inputs):
        w.weight[n] -= learning_rate * grads[n]

    # update bias
    w.bias -= learning_rate * grads[-1]

    print(f"loss after: {calculate_loss_mul(w, dataset)}")

def train_for_epochs(w, epochs, dataset,  fun, learning_rate = 1):
    for n in range(epochs):
        fun(w, dataset, learning_rate)
    return w

def main():
    AND_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    OR_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    
    model_AND = PerceptronMul(2)

    model_OR = PerceptronMul(2)

    train_for_epochs(model_AND, 100000, AND_dataset, single_step, 0.01,)
    train_for_epochs(model_OR, 100000, OR_dataset, single_step, 0.01)

    # tests
    print(f"AND for 1 and 1: {model_AND.return_result([1, 1])}")
    print(f"AND for 0 and 1: {model_AND.return_result([0, 1])}")
    print(f"AND for 1 and 0: {model_AND.return_result([1, 0])}")
    print(f"AND for 0 and 0: {model_AND.return_result([0, 0])}")

    print(f"OR for 1 and 1: {model_OR.return_result([1, 1])}")
    print(f"OR for 0 and 1: {model_OR.return_result([0, 1])}")
    print(f"OR for 1 and 0: {model_OR.return_result([1, 0])}")
    print(f"OR for 0 and 0: {model_OR.return_result([0, 0])}")

    # Now the values are much closer to the expected 
if __name__ == "__main__":
    main()