import numpy as np
try:
    from . import task01
except ImportError:
    import task01

def calculate_loss(w, dataset) -> float:
    sum = 0
    print(w)
    for (x, y) in dataset:
        print(sum)
        sum += (x * w - y)**2
    
    return sum / len(dataset)

def main():
    rng = np.random.default_rng(42)
    dataset = task01.create_dataset(6)
    w = rng.uniform(0, 10)

    loss = calculate_loss(w, dataset)
    print(f'MSE: {loss}') 

if __name__ == '__main__':
    main()