import matplotlib.pyplot as plt
import numpy as np

# https://www.geeksforgeeks.org/python/implement-sigmoid-function-using-numpy/
def main():
    x = np.linspace(-10, 10, 100)
    z = 1/(1 + np.exp(-x))

    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("Sigmoid(X)")

    plt.show()

if __name__ == "__main__":
    main()