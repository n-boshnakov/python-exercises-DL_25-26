import matplotlib.pyplot as plt
try:
    from . import task01, task06, task07
except ImportError:
    import task01, task06, task07


class PerceptronSigmoid():
    def __init__(self, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.bias = task01.initialize_weights(0, 1)
        self.loss_list = []

    def calc_loss(self, dataset, eps=0):
        sum = 0
        for (x, y, z) in dataset:
            sum += (task07.sigmoid((x * (self.w1 + eps) + y * (self.w2 + eps)) + self.bias) - z)**2
    
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
        sum = 0
        sum += self.w1*input1 + self.bias
        sum += self.w2*input2 + self.bias
        return (task07.sigmoid((self.w1*input1 + self.w2*input2) + self.bias))
    
    def return_loss_list(self):
        return self.loss_list

def main():
    dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    epochs = 100000
    learning_rate = 0.001

    perceptron_AND_bias = task06.PerceptronBias()
    perceptron_OR_bias = task06.PerceptronBias()
    
    perceptron_AND_bias.train(epochs, dataset_AND, learning_rate)
    perceptron_OR_bias.train(epochs, dataset_OR, learning_rate)


    perceptron_AND_sigmoid = task06.PerceptronBias()
    perceptron_OR_sigmoid = task06.PerceptronBias()
    
    perceptron_AND_sigmoid.train(epochs, dataset_AND, learning_rate)
    perceptron_OR_sigmoid.train(epochs, dataset_OR, learning_rate)

    # bias
    print(f"Bias - AND for 1 and 1: {perceptron_AND_bias.guess(1, 1)}")
    print(f"Bias - AND for 0 and 1: {perceptron_AND_bias.guess(0, 1)}")
    print(f"Bias - AND for 1 and 0: {perceptron_AND_bias.guess(1, 0)}")
    print(f"Bias - AND for 0 and 0: {perceptron_AND_bias.guess(0, 0)}")

    print(f"Bias - OR for 1 and 1: {perceptron_OR_bias.guess(1, 1)}")
    print(f"Bias - OR for 0 and 1: {perceptron_OR_bias.guess(0, 1)}")
    print(f"Bias - OR for 1 and 0: {perceptron_OR_bias.guess(1, 0)}")
    print(f"Bias - OR for 0 and 0: {perceptron_OR_bias.guess(0, 0)}")

    # sigmoid
    print(f"Sigmoid - AND for 1 and 1: {perceptron_AND_sigmoid.guess(1, 1)}")
    print(f"Sigmoid - AND for 0 and 1: {perceptron_AND_sigmoid.guess(0, 1)}")
    print(f"Sigmoid - AND for 1 and 0: {perceptron_AND_sigmoid.guess(1, 0)}")
    print(f"Sigmoid - AND for 0 and 0: {perceptron_AND_sigmoid.guess(0, 0)}")

    print(f"Sigmoid - OR for 1 and 1: {perceptron_OR_sigmoid.guess(1, 1)}")
    print(f"Sigmoid - OR for 0 and 1: {perceptron_OR_sigmoid.guess(0, 1)}")
    print(f"Sigmoid - OR for 1 and 0: {perceptron_OR_sigmoid.guess(1, 0)}")
    print(f"Sigmoid - OR for 0 and 0: {perceptron_OR_sigmoid.guess(0, 0)}")

    losses_list = [perceptron_AND_bias.return_loss_list(), perceptron_OR_bias.return_loss_list(), perceptron_AND_sigmoid.return_loss_list(), perceptron_OR_sigmoid.return_loss_list()]
    x = [i for i in range(0, len(losses_list[0]))]
    y = losses_list
    plt.xlabel("Epochs")
    plt.title("Loss")
    for i in range(len(y)):
        plt.plot(x, y[i])
    plt.show()

    # There appears to be a gradual decline in the loss that gradually reaches close to 0, but there is no big difference in how the loss gets reduced due to the implementation of the sigmoid
    
if __name__ == '__main__':
    main()