import numpy as np

try:
    from . import task05, task08
except ImportError:
    import task05, task08

class PerceptronMulSigmoidTest():
    def __init__(self, inputs):
        self.inputs = inputs
        self.weight = []
        self.bias = 0
        for n in range(self.inputs):
            self.weight.append(task05.initialize_weights(0, 1))

    def return_result(self, x):
        total_weight = 0
        for n in range(self.inputs):
            w = x[n] * self.weight[n]
            total_weight += w
        return total_weight + self.bias

def main():
    NAND_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    epochs = np.arange(start=0, stop=100000, step=1, dtype=None)

    model_NAND_sigmoid = PerceptronMulSigmoidTest(2)

    loss_hist_NAND = task08.train_for_epochs_loss_record(model_NAND_sigmoid, 100000, NAND_dataset, task08.single_step_sigmoid, 0.01)[1]

    # tests
    print(f"NAND for 1 and 1: {model_NAND_sigmoid.return_result([1, 1])}")
    print(f"NAND for 0 and 1: {model_NAND_sigmoid.return_result([0, 1])}")
    print(f"NAND for 1 and 0: {model_NAND_sigmoid.return_result([1, 0])}")
    print(f"NAND for 0 and 0: {model_NAND_sigmoid.return_result([0, 0])}")
    task08.plot(epochs, loss_hist_NAND)

    # using the regular weights and bias led to the perceptron learning to only return 1, no matter the input. I had to tweak the weights and bias to be a lot lower (from 1-10 to 0-1) and also lower the learning rate to 0.01 to get sort of passable results
if __name__ == "__main__":
    main()