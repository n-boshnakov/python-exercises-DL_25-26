import numpy as np
try:
    from . import task01, task03
except ImportError:
    import task01, task03

def run_for_epochs(num, w, dataset, learning_rate):
    w = 0
    for n in range(num):
        w = task03.single_step(w, dataset, learning_rate)
    return w


def main():
    # rng = np.random.default_rng(42)
    # w = rng.uniform(0, 10)
    
    dataset = task01.create_dataset(6)
    w = np.random.uniform(0, 10)

    w = run_for_epochs(500, w, dataset, 0.001)

    print(f"Final W: {w}")

if __name__ == '__main__':
    main()