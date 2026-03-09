import numpy as np

try:
    from . import task01, task02, task03
except ImportError:
    import task01, task02, task03

def main():
    dataset = task01.create_dataset(6)
    w = task02.Perceptron()
    task03.train_for_epochs(w, 500, dataset, 0.01)

if __name__ == '__main__':
    main()