import matplotlib.pyplot as plt
try:
    from . import task01, task06, task07, task08, task09
except ImportError:
    import task01, task06, task07, task08, task09


class ModelXOR():
    def __init__(self, w11=task01.initialize_weights(0, 10), w12=task01.initialize_weights(0, 10), w13=task01.initialize_weights(0, 10), w14=task01.initialize_weights(0, 10), w21=task01.initialize_weights(0, 10), w22=task01.initialize_weights(0, 10)):
        # self.w11 = w11
        # self.w12 = w12
        # self.w13 = w13
        # self.w14 = w14
        # self.w21 = w21
        # self.w22 = w22
        epochs = 100000
        learning_rate = 0.001

        dataset_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        dataset_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        dataset_NAND = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        self.perceptron_OR = task08.PerceptronSigmoid(dataset_OR)
        # self.perceptron_NAND = task08.PerceptronSigmoid(dataset_NAND)
        self.perceptron_NAND = task09.ModelNAND()
        self.perceptron_OR.train(epochs, learning_rate)
        # self.perceptron_NAND.train(epochs, learning_rate)
        self.perceptron_AND = task08.PerceptronSigmoid(dataset_AND)
        self.perceptron_AND.train(epochs, learning_rate)
        print(f"Sigmoid - AND for 1 and 1: {self.perceptron_AND.predict(1, 1)}")
        print(f"Sigmoid - AND for 0 and 1: {self.perceptron_AND.predict(0, 1)}")
        print(f"Sigmoid - AND for 1 and 0: {self.perceptron_AND.predict(1, 0)}")
        print(f"Sigmoid - AND for 0 and 0: {self.perceptron_AND.predict(0, 0)}")
        print("------------------------------------------------------------")
        print(f"Sigmoid - OR for 1 and 1: {self.perceptron_OR.predict(1, 1)}")
        print(f"Sigmoid - OR for 0 and 1: {self.perceptron_OR.predict(0, 1)}")
        print(f"Sigmoid - OR for 1 and 0: {self.perceptron_OR.predict(1, 0)}")
        print(f"Sigmoid - OR for 0 and 0: {self.perceptron_OR.predict(0, 0)}")
        print("------------------------------------------------------------")
        print(f"NAND for 1 and 1: {self.perceptron_NAND.predict(1, 1)}")
        print(f"NAND for 0 and 1: {self.perceptron_NAND.predict(0, 1)}")
        print(f"NAND for 1 and 0: {self.perceptron_NAND.predict(1, 0)}")
        print(f"NAND for 0 and 0: {self.perceptron_NAND.predict(0, 0)}")


    def predict(self, input1, input2):
        pred_OR = (1 if self.perceptron_OR.predict(input1, input2) >= 0.5 else 0)
        print(f"Pred_OR: {pred_OR}")
        pred_NAND = (1 if self.perceptron_NAND.predict(input1, input2) >= 0.5 else 0)
        print(f"Pred_NAND: {pred_NAND}")

        return (self.perceptron_AND.predict(pred_OR, pred_NAND))


def main():
    modelXOR = ModelXOR()

    # bias
    print(f"XOR for 1 and 1: {modelXOR.predict(1, 1)}")
    print(f"XOR for 0 and 1: {modelXOR.predict(0, 1)}")
    print(f"XOR for 1 and 0: {modelXOR.predict(1, 0)}")
    print(f"XOR for 0 and 0: {modelXOR.predict(0, 0)}")
    
if __name__ == '__main__':
    main()