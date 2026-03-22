import numpy as np
try:
    from . import task01
except ImportError:
    import task01

def calculate_loss(w, dataset) -> float:
    sum = 0
    for (x, y) in dataset:
        sum += (x * w - y)**2
    
    return sum / len(dataset)

def main():
    rng = np.random.default_rng(42)
    dataset = task01.create_dataset(6)
    w = rng.uniform(0, 10)

    loss = calculate_loss(w, dataset)
    # regular loss + 0.001 * 2: 302.18390408561453
    # regular loss + 0.01: 302.07865131004604
    # regular loss: 301.9734168678107
    # regular loss - 0.001: 301.86820075890876
    # regular loss - 0.001 * 2: 301.76300298334024

    # When increasing the weight, the loss increases; when decreasing the weight, the loss decreases; modifying the values that we increase/decrease by also modifies by how much the loss changes
    print(f'MSE: {loss}') 

if __name__ == '__main__':
    main()