import numpy as np
try:
    from . import task01, task02, task03, task04
except ImportError:
    import task01, task02, task03, task04


class PerceptronBias():
    def __init__(self, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.bias = task01.initialize_weights(0, 1)
        self.loss_list = []

    def calc_loss(self, dataset, eps=0):
        sum = 0
        for (x, y, z) in dataset:
            sum += (((x * (self.w1 + eps) + y * (self.w2 + eps)) + self.bias) - z)**2
    
        return sum / len(dataset)
    
    def calc_derivative(self, dataset, eps=0.01):
        loss1 = self.calc_loss(dataset)
        loss2 = self.calc_loss(dataset, eps)
        self.loss_list.append((loss2 - loss1) / eps)

        return (loss2 - loss1) / eps
    
    def single_step(self, dataset, learning_rate):
        print(f"Loss before: {self.calc_loss(dataset)}")
        derivative = self.calc_derivative(dataset)
        self.w1 -= derivative * learning_rate
        self.w2 -= derivative * learning_rate
        print(f"Loss after: {self.calc_loss(dataset)}")
        return self.w1, self.w2

    def train(self, epochs, dataset, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(dataset, learning_rate)
            pass

    def guess(self, input1, input2):
        return (self.w1*input1 + self.w2*input2 + self.bias)

    def return_loss_list(self):
        return self.loss_list

def main():
    dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    epochs = 100000
    learning_rate = 0.001

    perceptron_AND = PerceptronBias()
    perceptron_OR = PerceptronBias()
    
    perceptron_AND.train(epochs, dataset_AND, learning_rate)
    perceptron_OR.train(epochs, dataset_OR, learning_rate)

    print(f"AND for 1 and 1: {perceptron_AND.guess(1, 1)}")
    print(f"AND for 0 and 1: {perceptron_AND.guess(0, 1)}")
    print(f"AND for 1 and 0: {perceptron_AND.guess(1, 0)}")
    print(f"AND for 0 and 0: {perceptron_AND.guess(0, 0)}")

    print(f"OR for 1 and 1: {perceptron_OR.guess(1, 1)}")
    print(f"OR for 0 and 1: {perceptron_OR.guess(0, 1)}")
    print(f"OR for 1 and 0: {perceptron_OR.guess(1, 0)}")
    print(f"OR for 0 and 0: {perceptron_OR.guess(0, 0)}")

    # What do you notice about the confidence the models have in their predicted values?
    # There appears to be an issue with learning, depending on which number is first - 0 or 1. If it's 0, then the answer is always less than 1, if it's 1 - positive.

if __name__ == '__main__':
    main()