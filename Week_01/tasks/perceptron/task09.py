import matplotlib.pyplot as plt
try:
    from . import task01, task06, task07, task08
except ImportError:
    import task01, task06, task07, task08


class ModelNAND():
    def __init__(self, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10), w3=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        epochs = 100000
        learning_rate = 0.001

        dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        dataset_NOT = [(0, 1), (1, 0)]
        self.perceptron_AND = task08.PerceptronSigmoid(dataset_AND, self.w1, self.w2)
        self.perceptron_AND.train(epochs, learning_rate)
        self.perceptron_NOT = PerceptronSigmoidSingle(dataset_NOT, self.w3)
        self.perceptron_NOT.train(epochs, learning_rate)

    def predict(self, input1, input2):

        pred_AND = (1 if self.perceptron_AND.predict(input1, input2) >= 0.5 else 0)

        return (self.perceptron_NOT.predict(pred_AND))

class PerceptronSigmoidSingle():
    def __init__(self, dataset, w1=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.dataset = dataset
        self.bias = task01.initialize_weights(0, 1)
        self.loss_list = []

    def calc_loss(self, w1_eps: float=0, b_eps: float=0):
        sum = 0
        for (x, z) in self.dataset:
            sum += (task07.sigmoid((x * (self.w1 + w1_eps)) + (self.bias + b_eps)) - z)**2
    
        return sum / len(self.dataset)
    
    def calc_derivative(self, w1_eps: float=0, b_eps: float=0):
        loss1 = self.calc_loss()
        loss2 = self.calc_loss(w1_eps, b_eps)
        self.loss_list.append((loss2 - loss1) / (w1_eps if w1_eps != 0 else b_eps))

        return (loss2 - loss1) / (w1_eps if w1_eps != 0 else b_eps)
    
    def single_step(self, learning_rate):
        print(f"Loss before: {self.calc_loss()}")
        d_w1 = self.calc_derivative(w1_eps = 0.001)
        d_b = self.calc_derivative(b_eps = 0.001)

        self.w1 -= d_w1 * learning_rate
        self.bias -= d_b * learning_rate
        print(f"Loss after: {self.calc_loss()}")
        return self.w1

    def train(self, epochs, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(learning_rate)
            pass

    def predict(self, input1):
        return (task07.sigmoid((self.w1*input1) + self.bias))

def main():
    modelNAND = ModelNAND()

    # testing NOT perceptron
    print(f"NOT for 1: {modelNAND.perceptron_NOT.predict(1)}")
    print(f"NOT for 0: {modelNAND.perceptron_NOT.predict(0)}")

    # NAND Model results
    print(f"NAND for 1 and 1: {modelNAND.predict(1, 1)}")
    print(f"NAND for 0 and 1: {modelNAND.predict(0, 1)}")
    print(f"NAND for 1 and 0: {modelNAND.predict(1, 0)}")
    print(f"NAND for 0 and 0: {modelNAND.predict(0, 0)}")
    
if __name__ == '__main__':
    main()


# Previous solution, reusing the perceptron class created for AND and OR; Did not work
# import numpy as np
# try:
#     from . import task01, task06, task07, task08
# except ImportError:
#     import task01, task06, task07, task08



# def main():
#     NAND_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
#     epochs = 100000
#     learning_rate = 0.001

#     # This should have worked, no idea why it didn't. I tried to solve the task by building a multi-layer perceptron (https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d) but hit a wall with the implementation
#     perceptron_NAND = task08.PerceptronSigmoid(NAND_dataset)

#     perceptron_NAND.train(epochs, learning_rate)

#     # tests
#     print(f"NAND for 1 and 1: {perceptron_NAND.predict(1, 1)}")
#     print(f"NAND for 0 and 1: {perceptron_NAND.predict(0, 1)}")
#     print(f"NAND for 1 and 0: {perceptron_NAND.predict(1, 0)}")
#     print(f"NAND for 0 and 0: {perceptron_NAND.predict(0, 0)}")

#     # Returns:
#     # NAND for 1 and 1: 0.999999994046725
#     # NAND for 0 and 1: 0.999900929781145
#     # NAND for 1 and 0: 0.999991780877754
#     # NAND for 0 and 0: 0.8796687605047867

# if __name__ == '__main__':
#     main()