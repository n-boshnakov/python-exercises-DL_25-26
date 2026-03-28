import matplotlib.pyplot as plt
try:
    from . import task01, task06, task07
except ImportError:
    import task01, task06, task07


class PerceptronSigmoid():
    def __init__(self, dataset, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.dataset = dataset
        self.bias = task01.initialize_weights(0, 1)
        self.loss_list = []

    def calc_loss(self, w1_eps: float=0, w2_eps: float=0, b_eps: float=0):
        sum = 0
        for (x, y, z) in self.dataset:
            sum += (task07.sigmoid((x * (self.w1 + w1_eps) + y * (self.w2 + w2_eps)) + (self.bias + b_eps)) - z)**2
    
        return sum / len(self.dataset)
    
    def calc_derivative(self, w1_eps: float=0, w2_eps: float=0, b_eps: float=0):
        loss1 = self.calc_loss()
        loss2 = self.calc_loss(w1_eps, w2_eps, b_eps)
        self.loss_list.append((loss2 - loss1) / (w1_eps if w1_eps != 0 else w2_eps if w2_eps != 0 else b_eps))

        return (loss2 - loss1) / (w1_eps if w1_eps != 0 else w2_eps if w2_eps != 0 else b_eps)
    
    def single_step(self, learning_rate):
        print(f"Loss before: {self.calc_loss()}")
        d_w1 = self.calc_derivative(w1_eps = 0.001)
        d_w2 = self.calc_derivative(w2_eps = 0.001)
        d_b = self.calc_derivative(b_eps = 0.001)

        self.w1 -= d_w1 * learning_rate
        self.w2 -= d_w2 * learning_rate
        self.bias -= d_b * learning_rate
        print(f"Loss after: {self.calc_loss()}")
        return self.w1, self.w2

    def train(self, epochs, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(learning_rate)
            pass

    def predict(self, input1, input2):
        return (task07.sigmoid((self.w1*input1 + self.w2*input2) + self.bias))
    
    def return_loss_list(self):
        return self.loss_list

def main():
    dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    epochs = 100
    learning_rate = 0.001

    perceptron_AND_bias = task06.PerceptronBias(dataset_AND)
    perceptron_OR_bias = task06.PerceptronBias(dataset_OR)
    
    perceptron_AND_bias.train(epochs, learning_rate)
    perceptron_OR_bias.train(epochs, learning_rate)


    perceptron_AND_sigmoid = PerceptronSigmoid(dataset_AND)
    perceptron_OR_sigmoid = PerceptronSigmoid(dataset_OR)
    
    perceptron_AND_sigmoid.train(epochs, learning_rate)
    perceptron_OR_sigmoid.train(epochs, learning_rate)

    # bias
    print(f"Bias - AND for 1 and 1: {perceptron_AND_bias.predict(1, 1)}")
    print(f"Bias - AND for 0 and 1: {perceptron_AND_bias.predict(0, 1)}")
    print(f"Bias - AND for 1 and 0: {perceptron_AND_bias.predict(1, 0)}")
    print(f"Bias - AND for 0 and 0: {perceptron_AND_bias.predict(0, 0)}")

    print(f"Bias - OR for 1 and 1: {perceptron_OR_bias.predict(1, 1)}")
    print(f"Bias - OR for 0 and 1: {perceptron_OR_bias.predict(0, 1)}")
    print(f"Bias - OR for 1 and 0: {perceptron_OR_bias.predict(1, 0)}")
    print(f"Bias - OR for 0 and 0: {perceptron_OR_bias.predict(0, 0)}")

    # sigmoid
    print(f"Sigmoid - AND for 1 and 1: {perceptron_AND_sigmoid.predict(1, 1)}")
    print(f"Sigmoid - AND for 0 and 1: {perceptron_AND_sigmoid.predict(0, 1)}")
    print(f"Sigmoid - AND for 1 and 0: {perceptron_AND_sigmoid.predict(1, 0)}")
    print(f"Sigmoid - AND for 0 and 0: {perceptron_AND_sigmoid.predict(0, 0)}")

    print(f"Sigmoid - OR for 1 and 1: {perceptron_OR_sigmoid.predict(1, 1)}")
    print(f"Sigmoid - OR for 0 and 1: {perceptron_OR_sigmoid.predict(0, 1)}")
    print(f"Sigmoid - OR for 1 and 0: {perceptron_OR_sigmoid.predict(1, 0)}")
    print(f"Sigmoid - OR for 0 and 0: {perceptron_OR_sigmoid.predict(0, 0)}")

    losses_list = [perceptron_AND_bias.return_loss_list(), perceptron_OR_bias.return_loss_list(), perceptron_AND_sigmoid.return_loss_list(), perceptron_OR_sigmoid.return_loss_list()]
    x = [i for i in range(0, len(losses_list[0]))]
    y = losses_list
    plt.xlabel("Epochs")
    plt.title("Loss")
    for i in range(len(y)):
        plt.plot(x, y[i])
    plt.show()

    # The sigmoid makes the predictions much closer to the expected values; 
    # The loss plot shows the gradual decline and eventual reaching of the final values for the weights;
    # Due to the sigmoid function, the values for the sigmoid perceptrons are much smaller 
    
if __name__ == '__main__':
    main()