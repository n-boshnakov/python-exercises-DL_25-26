import numpy as np
try:
    from . import task01
except ImportError:
    import task01

def calculate_loss(w, dataset):
    MSE = 0
    for data in dataset:
        curr_e = (data[1] - w.return_result(data[0])) ** 2
        MSE += curr_e
    return MSE / len(dataset)

class Perceptron():
    def __init__(self):
        self.weight = task01.initialize_weights(0, 10)

    def return_result(self, x):
        return x * self.weight
    
def main():
    dataset = task01.create_dataset(6)
    w = Perceptron()

    loss = calculate_loss(w, dataset)

    print(f'MSE: {loss}')

if __name__ == '__main__':
    main()