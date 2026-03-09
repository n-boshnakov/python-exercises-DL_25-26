import numpy as np
try:
    from . import task01, task02
except ImportError:
    import task01, task02

# function created with help from Copilot
def finite_difference_derivative(w, dataset, eps=1e-5):
    loss1 = task02.calculate_loss(w, dataset)

    # temporarily modify the weight
    w.weight += eps
    loss2 = task02.calculate_loss(w, dataset)

    # restore the weight
    w.weight -= eps

    return (loss2 - loss1) / eps

def single_step(w, dataset, learning_rate=1):
    print(f"loss before: {task02.calculate_loss(w, dataset)}")
    w.weight -= learning_rate * finite_difference_derivative(w, dataset)
    print(f"loss after: {task02.calculate_loss(w, dataset)}")

def train_for_epochs(w, epochs, dataset, learning_rate = 1):
    for n in range(epochs):
        single_step(w, dataset, learning_rate)
    return w

def main():
    dataset = task01.create_dataset(6)
    w = task02.Perceptron()
# task 03; testing without a learning_rate (equal to 1)
    train_for_epochs(w, 10, dataset)

    # implementing a test for adding a learning rate -> learning slows down drastically
    w1 = task02.Perceptron()
    print("Test for learning_rate = 0.001")
    train_for_epochs(w1, 10, dataset, 0.001)

    print(
        f"After training for 10 epochs with learning_rate = 0.001: 5*2 = {w1.return_result(5)}"
    )

    # implementing a test for adding a bigger learning rate -> learning is much better
    w2 = task02.Perceptron()
    print("Test for learning_rate = 0.01")
    train_for_epochs(w2, 10, dataset, 0.01)
    print(
        f"After training for 10 epochs with learning_rate = 0.01: 5*2 = {w2.return_result(5)}"
    )

if __name__ == '__main__':
    main()