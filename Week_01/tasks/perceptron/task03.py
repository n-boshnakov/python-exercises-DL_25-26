import numpy as np
try:
    from . import task01, task02
except ImportError:
    import task01, task02


def calculate_derivative(w, dataset, eps=0.01) -> float:
    loss1 = task02.calculate_loss(w, dataset)
    loss2 = task02.calculate_loss(w + eps, dataset)

    print(f"Loss after: {(loss2 - loss1) / eps}")

    return (loss2 - loss1) / eps

def single_step(w, dataset, learning_rate=1):
    print(f"Loss before: {task02.calculate_loss(w, dataset)}")
    derivative = calculate_derivative(w, dataset)
    w -= derivative * learning_rate
    print(f"Loss after: {task02.calculate_loss(w, dataset)}")
    return w


def main():
    rng = np.random.default_rng(42)
    dataset = task01.create_dataset(6)
    w = rng.uniform(0, 10)

    calculate_derivative(w, dataset)
    print(w)

    # with no learning_rate (equal to 1)
    # results are abysmal, the model overfits and doesn't learn anything
    for n in range(10):
        w = single_step(w, dataset)

    # with learning_rate of 0.001
    # after 10 iterations, the model is now closer to the answer (~6.7 vs ~7.7 originally)
    for n in range(10):
        w = single_step(w, dataset, 0.001)

    print(w)

if __name__ == '__main__':
    main()