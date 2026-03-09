import numpy as np

np.random.seed(42)

# The model has 1 parameters -> the single weight initialized by initialize_weights(x, y)

def create_dataset(n):
    results_list = []
    for n in range(n):
        a = n
        b = n * 2
        results_list.append((a, b))
    return results_list


def initialize_weights(x, y):
    return np.random.uniform(x, y)


def main():
    dataset = create_dataset(6)
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))
    print(dataset)


if __name__ == '__main__':
    main()
