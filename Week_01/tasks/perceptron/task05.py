try:
    from . import task01
except ImportError:
    import task01


class Perceptron():
    def __init__(self, dataset, w1=task01.initialize_weights(0, 10), w2=task01.initialize_weights(0, 10)):
        self.w1 = w1
        self.w2 = w2
        self.dataset = dataset

    def calc_loss(self, w1_eps: float=0, w2_eps: float=0):
        sum = 0
        for (x, y, z) in self.dataset:
            sum += ((x * (self.w1 + w1_eps) + y * (self.w2 + w2_eps)) - z)**2
    
        return sum / len(self.dataset)
    
    def calc_derivative(self, w1_eps: float=0, w2_eps: float=0):
        loss1 = self.calc_loss()
        loss2 = self.calc_loss(w1_eps, w2_eps)

        return (loss2 - loss1) / (w1_eps if w1_eps != 0 else w2_eps)
    
    def single_step(self, learning_rate):
        print(f"Loss before: {self.calc_loss()}")

        dw1 = self.calc_derivative(w1_eps=0.01)
        dw2 = self.calc_derivative(w2_eps=0.01)

        self.w1 -= dw1 * learning_rate
        self.w2 -= dw2 * learning_rate

        print(f"Loss after: {self.calc_loss()}")
        return self.w1, self.w2

    def train(self, epochs, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(learning_rate)

    def predict(self, input1, input2):
        return (self.w1*input1 + self.w2*input2)

def main():
    dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    epochs = 100000
    learning_rate = 0.001

    perceptron_AND = Perceptron(dataset_AND)
    perceptron_OR = Perceptron(dataset_OR)
    
    print(perceptron_AND.w1)
    print(perceptron_AND.w2)
    perceptron_AND.train(epochs, learning_rate)
    perceptron_OR.train(epochs, learning_rate)
    # General forms of the two models:
    # each model has 2 parameters - its weights (one for each input)

    print(f"AND for 1 and 1: {perceptron_AND.predict(1, 1)}")
    print(f"AND for 0 and 1: {perceptron_AND.predict(0, 1)}")
    print(f"AND for 1 and 0: {perceptron_AND.predict(1, 0)}")
    print(f"AND for 0 and 0: {perceptron_AND.predict(0, 0)}")

    print(f"OR for 1 and 1: {perceptron_OR.predict(1, 1)}")
    print(f"OR for 0 and 1: {perceptron_OR.predict(0, 1)}")
    print(f"OR for 1 and 0: {perceptron_OR.predict(1, 0)}")
    print(f"OR for 0 and 0: {perceptron_OR.predict(0, 0)}")

    # What do you notice about the confidence the models have in their predicted values?
    # When it comes to the AND perceptron, it has learned lower values overall (1 and 1 returns 0.65), while the OR perceptron has learned higher overall values (1 and 1 returns 1.32)

if __name__ == '__main__':
    main()