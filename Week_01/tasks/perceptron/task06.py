import numpy as np
try:
    from . import task01, task02, task03, task04
except ImportError:
    import task01, task02, task03, task04


class PerceptronBias():
    def __init__(self, dataset, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.dataset = dataset
        self.bias = task01.initialize_weights(0, 1)
        self.loss_list = []

    def calc_loss(self, eps_w1: float=0, eps_w2: float=0, eps_b: float=0):
        sum = 0
        for (x, y, z) in self.dataset:
            sum += (((x * (self.w1 + eps_w1) + y * (self.w2 + eps_w2)) + (self.bias + eps_b) - z)**2)
    
        return sum / len(self.dataset)
    
    def calc_derivative(self, eps_w1: float=0, eps_w2: float=0, eps_b: float=0):
        loss1 = self.calc_loss()
        loss2 = self.calc_loss(eps_w1, eps_w2, eps_b)
        self.loss_list.append((loss2 - loss1) / (eps_w1 if eps_w1 != 0 else eps_w2 if eps_w2 != 0 else eps_b))

        return (loss2 - loss1) / (eps_w1 if eps_w1 != 0 else eps_w2 if eps_w2 != 0 else eps_b)
    
    def single_step(self, learning_rate):
        print(f"Loss before: {self.calc_loss()}")
        dw1 = self.calc_derivative(eps_w1 = 0.001)
        dw2 = self.calc_derivative(eps_w2 = 0.001)
        db = self.calc_derivative(eps_b = 0.001)

        self.w1 -= dw1 * learning_rate
        self.w2 -= dw2 * learning_rate
        self.bias -= db * learning_rate

        print(f"Loss after: {self.calc_loss()}")
        return self.w1, self.w2

    def train(self, epochs, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(learning_rate)

    def predict(self, input1, input2):
        return (self.w1*input1 + self.w2*input2 + self.bias)

    def return_loss_list(self):
        return self.loss_list

def main():
    dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    epochs = 100000
    learning_rate = 0.001

    perceptron_AND = PerceptronBias(dataset_AND)
    perceptron_OR = PerceptronBias(dataset_OR)
    
    perceptron_AND.train(epochs, learning_rate)
    perceptron_OR.train(epochs, learning_rate)

    print(f"AND for 1 and 1: {perceptron_AND.predict(1, 1)}")
    print(f"AND for 0 and 1: {perceptron_AND.predict(0, 1)}")
    print(f"AND for 1 and 0: {perceptron_AND.predict(1, 0)}")
    print(f"AND for 0 and 0: {perceptron_AND.predict(0, 0)}")

    print(f"OR for 1 and 1: {perceptron_OR.predict(1, 1)}")
    print(f"OR for 0 and 1: {perceptron_OR.predict(0, 1)}")
    print(f"OR for 1 and 0: {perceptron_OR.predict(1, 0)}")
    print(f"OR for 0 and 0: {perceptron_OR.predict(0, 0)}")

    # What has changed in comparison to the previous task?
    # The values are now closer to their expected values than without the bias

if __name__ == '__main__':
    main()