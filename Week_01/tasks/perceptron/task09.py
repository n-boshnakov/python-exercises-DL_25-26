import numpy as np
try:
    from . import task01, task08
except ImportError:
    import task01, task08



def main():
    NAND_dataset = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    epochs = 100000
    learning_rate = 0.01

    # This should have worked, no idea why it didn't. I tried to solve the task by building a multi-layer perceptron (https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d) but hit a wall with the implementation
    perceptron_NAND = task08.PerceptronSigmoid(NAND_dataset, task01.initialize_weights(-1, 1), task01.initialize_weights(-1, 1))

    perceptron_NAND.train(epochs, learning_rate)

    # tests
    print(f"NAND for 1 and 1: {perceptron_NAND.predict(1, 1)}")
    print(f"NAND for 0 and 1: {perceptron_NAND.predict(0, 1)}")
    print(f"NAND for 1 and 0: {perceptron_NAND.predict(1, 0)}")
    print(f"NAND for 0 and 0: {perceptron_NAND.predict(0, 0)}")

    # Returns:
    # NAND for 0 and 1: 0.9177018823301617
    # NAND for 1 and 0: 0.9177018983853135
    # NAND for 0 and 0: 0.9991251477290325
    # NAND for 1 and 1: 0.09818717680303782

if __name__ == '__main__':
    main()