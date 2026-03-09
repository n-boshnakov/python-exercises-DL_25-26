import numpy as np

try:
    from . import task08, task09
except ImportError:
    import task08, task09

class Xor():
    def __init__(self):
        self.AND_perceptron = task09.PerceptronMulSigmoidTest(2)
        self.OR_perceptron = task09.PerceptronMulSigmoidTest(2)
        self.NAND_perceptron = task09.PerceptronMulSigmoidTest(2)

    def binarize(self, x):
        return 1 if x >= 0.5 else 0

    def forward(self, model, x, y):
        return model.return_result([x, y])
    
    def return_result(self, x, y):
        res_NAND = self.binarize(self.forward(self.NAND_perceptron, x, y))
        res_OR = self.binarize(self.forward(self.OR_perceptron, x, y))
        res_AND = self.binarize(self.forward(self.AND_perceptron, res_NAND, res_OR))
        return res_AND


def main():
    AND_dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    OR_dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    NAND_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    model_XOR = Xor()
    task08.train_for_epochs_loss_record(model_XOR.AND_perceptron, 100000, AND_dataset, task08.single_step_sigmoid, 0.01)
    task08.train_for_epochs_loss_record(model_XOR.OR_perceptron, 10000, OR_dataset, task08.single_step_sigmoid, 0.01)
    task08.train_for_epochs_loss_record(model_XOR.NAND_perceptron, 100000, NAND_dataset, task08.single_step_sigmoid, 0.01)

    print(f"XOR for 1 and 1: {model_XOR.return_result(1, 1)}")
    print(f"XOR for 0 and 1: {model_XOR.return_result(0, 1)}")
    print(f"XOR for 1 and 0: {model_XOR.return_result(1, 0)}")
    print(f"XOR for 0 and 0: {model_XOR.return_result(0, 0)}")

if __name__ == "__main__":
    main()