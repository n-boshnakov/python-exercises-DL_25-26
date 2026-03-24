import matplotlib.pyplot as plt
import numpy as np

# https://www.geeksforgeeks.org/python/implement-sigmoid-function-using-numpy/
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    x = np.linspace(-10, 10, 100)
    z = sigmoid(x)

    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("Sigmoid(X)")

    plt.show()

if __name__ == "__main__":
    main()