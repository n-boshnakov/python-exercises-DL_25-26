import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import copy
import random

try:
    from . import task01, task06, task07, task08
except ImportError:
    import task01, task06, task07, task08

class ModelSquare:

    def __init__(self) -> None:
        # Input Layer: 1 -> 8
        self.l1_w1 = task01.initialize_weights(-1, 1)
        self.l1_b1 = task01.initialize_weights(-1, 1)

        self.l1_w2 = task01.initialize_weights(-1, 1)
        self.l1_b2 = task01.initialize_weights(-1, 1)

        self.l1_w3 = task01.initialize_weights(-1, 1)
        self.l1_b3 = task01.initialize_weights(-1, 1)

        self.l1_w4 = task01.initialize_weights(-1, 1)
        self.l1_b4 = task01.initialize_weights(-1, 1)

        self.l1_w5 = task01.initialize_weights(-1, 1)
        self.l1_b5 = task01.initialize_weights(-1, 1)

        self.l1_w6 = task01.initialize_weights(-1, 1)
        self.l1_b6 = task01.initialize_weights(-1, 1)

        self.l1_w7 = task01.initialize_weights(-1, 1)
        self.l1_b7 = task01.initialize_weights(-1, 1)

        self.l1_w8 = task01.initialize_weights(-1, 1)
        self.l1_b8 = task01.initialize_weights(-1, 1)

        # Hidden Layer: 8 -> 8

        self.l2_w11 =  task01.initialize_weights(-1, 1)
        self.l2_w12 =  task01.initialize_weights(-1, 1)
        self.l2_w13 =  task01.initialize_weights(-1, 1)
        self.l2_w14 =  task01.initialize_weights(-1, 1)
        self.l2_w15 =  task01.initialize_weights(-1, 1)
        self.l2_w16 =  task01.initialize_weights(-1, 1)
        self.l2_w17 =  task01.initialize_weights(-1, 1)
        self.l2_w18 =  task01.initialize_weights(-1, 1)
        self.l2_b1 =  task01.initialize_weights(-1, 1)

        self.l2_w21 =  task01.initialize_weights(-1, 1)
        self.l2_w22 =  task01.initialize_weights(-1, 1)
        self.l2_w23 =  task01.initialize_weights(-1, 1)
        self.l2_w24 =  task01.initialize_weights(-1, 1)
        self.l2_w25 =  task01.initialize_weights(-1, 1)
        self.l2_w26 =  task01.initialize_weights(-1, 1)
        self.l2_w27 =  task01.initialize_weights(-1, 1)
        self.l2_w28 =  task01.initialize_weights(-1, 1)
        self.l2_b2 =  task01.initialize_weights(-1, 1)

        self.l2_w31 =  task01.initialize_weights(-1, 1)
        self.l2_w32 =  task01.initialize_weights(-1, 1)
        self.l2_w33 =  task01.initialize_weights(-1, 1)
        self.l2_w34 =  task01.initialize_weights(-1, 1)
        self.l2_w35 =  task01.initialize_weights(-1, 1)
        self.l2_w36 =  task01.initialize_weights(-1, 1)
        self.l2_w37 =  task01.initialize_weights(-1, 1)
        self.l2_w38 =  task01.initialize_weights(-1, 1)
        self.l2_b3 =  task01.initialize_weights(-1, 1)

        self.l2_w41 =  task01.initialize_weights(-1, 1)
        self.l2_w42 =  task01.initialize_weights(-1, 1)
        self.l2_w43 =  task01.initialize_weights(-1, 1)
        self.l2_w44 =  task01.initialize_weights(-1, 1)
        self.l2_w45 =  task01.initialize_weights(-1, 1)
        self.l2_w46 =  task01.initialize_weights(-1, 1)
        self.l2_w47 =  task01.initialize_weights(-1, 1)
        self.l2_w48 =  task01.initialize_weights(-1, 1)
        self.l2_b4 =  task01.initialize_weights(-1, 1)

        self.l2_w51 =  task01.initialize_weights(-1, 1)
        self.l2_w52 =  task01.initialize_weights(-1, 1)
        self.l2_w53 =  task01.initialize_weights(-1, 1)
        self.l2_w54 =  task01.initialize_weights(-1, 1)
        self.l2_w55 =  task01.initialize_weights(-1, 1)
        self.l2_w56 =  task01.initialize_weights(-1, 1)
        self.l2_w57 =  task01.initialize_weights(-1, 1)
        self.l2_w58 =  task01.initialize_weights(-1, 1)
        self.l2_b5 =  task01.initialize_weights(-1, 1)

        self.l2_w61 =  task01.initialize_weights(-1, 1)
        self.l2_w62 =  task01.initialize_weights(-1, 1)
        self.l2_w63 =  task01.initialize_weights(-1, 1)
        self.l2_w64 =  task01.initialize_weights(-1, 1)
        self.l2_w65 =  task01.initialize_weights(-1, 1)
        self.l2_w66 =  task01.initialize_weights(-1, 1)
        self.l2_w67 =  task01.initialize_weights(-1, 1)
        self.l2_w68 =  task01.initialize_weights(-1, 1)
        self.l2_b6 =  task01.initialize_weights(-1, 1)

        self.l2_w71 =  task01.initialize_weights(-1, 1)
        self.l2_w72 =  task01.initialize_weights(-1, 1)
        self.l2_w73 =  task01.initialize_weights(-1, 1)
        self.l2_w74 =  task01.initialize_weights(-1, 1)
        self.l2_w75 =  task01.initialize_weights(-1, 1)
        self.l2_w76 =  task01.initialize_weights(-1, 1)
        self.l2_w77 =  task01.initialize_weights(-1, 1)
        self.l2_w78 =  task01.initialize_weights(-1, 1)
        self.l2_b7 =  task01.initialize_weights(-1, 1)

        self.l2_w81 =  task01.initialize_weights(-1, 1)
        self.l2_w82 =  task01.initialize_weights(-1, 1)
        self.l2_w83 =  task01.initialize_weights(-1, 1)
        self.l2_w84 =  task01.initialize_weights(-1, 1)
        self.l2_w85 =  task01.initialize_weights(-1, 1)
        self.l2_w86 =  task01.initialize_weights(-1, 1)
        self.l2_w87 =  task01.initialize_weights(-1, 1)
        self.l2_w88 =  task01.initialize_weights(-1, 1)
        self.l2_b8 =  task01.initialize_weights(-1, 1)

        # Output Layer: 8 -> 1

        self.l3_w1 =  task01.initialize_weights(-1, 1)
        self.l3_w2 =  task01.initialize_weights(-1, 1)
        self.l3_w3 =  task01.initialize_weights(-1, 1)
        self.l3_w4 =  task01.initialize_weights(-1, 1)
        self.l3_w5 =  task01.initialize_weights(-1, 1)
        self.l3_w6 =  task01.initialize_weights(-1, 1)
        self.l3_w7 =  task01.initialize_weights(-1, 1)
        self.l3_w8 =  task01.initialize_weights(-1, 1)
        self.l3_b =  task01.initialize_weights(-1, 1)


    def parameter_names(self) -> list[str]:
        return [
            "l1_w1", "l1_b1", "l1_w2", "l1_b2", "l1_w3", "l1_b3", "l1_w4", "l1_b4",
            "l1_w5", "l1_b5", "l1_w6", "l1_b6", "l1_w7", "l1_b7", "l1_w8", "l1_b8",

            "l2_w11", "l2_w12", "l2_w13", "l2_w14", "l2_w15", "l2_w16", "l2_w17", "l2_w18", "l2_b1",
            "l2_w21", "l2_w22", "l2_w23", "l2_w24", "l2_w25", "l2_w26", "l2_w27", "l2_w28", "l2_b2",
            "l2_w31", "l2_w32", "l2_w33", "l2_w34", "l2_w35", "l2_w36", "l2_w37", "l2_w38", "l2_b3",
            "l2_w41", "l2_w42", "l2_w43", "l2_w44", "l2_w45", "l2_w46", "l2_w47", "l2_w48", "l2_b4",
            "l2_w51", "l2_w52", "l2_w53", "l2_w54", "l2_w55", "l2_w56", "l2_w57", "l2_w58", "l2_b5",
            "l2_w61", "l2_w62", "l2_w63", "l2_w64", "l2_w65", "l2_w66", "l2_w67", "l2_w68", "l2_b6",
            "l2_w71", "l2_w72", "l2_w73", "l2_w74", "l2_w75", "l2_w76", "l2_w77", "l2_w78", "l2_b7",
            "l2_w81", "l2_w82", "l2_w83", "l2_w84", "l2_w85", "l2_w86", "l2_w87", "l2_w88", "l2_b8",

            "l3_w1", "l3_w2", "l3_w3", "l3_w4", "l3_w5", "l3_w6", "l3_w7", "l3_w8", "l3_b",
        ]

    def forward(self, x: float) -> float:
        h1 = torch.relu(torch.tensor(self.l1_w1 * x + self.l1_b1, dtype=torch.float32)).item()
        h2 = torch.relu(torch.tensor(self.l1_w2 * x + self.l1_b1, dtype=torch.float32)).item()
        h3 = torch.relu(torch.tensor(self.l1_w3 * x + self.l1_b1, dtype=torch.float32)).item()
        h4 = torch.relu(torch.tensor(self.l1_w4 * x + self.l1_b4, dtype=torch.float32)).item()
        h5 = torch.relu(torch.tensor(self.l1_w5 * x + self.l1_b5, dtype=torch.float32)).item()
        h6 = torch.relu(torch.tensor(self.l1_w6 * x + self.l1_b6, dtype=torch.float32)).item()
        h7 = torch.relu(torch.tensor(self.l1_w7 * x + self.l1_b7, dtype=torch.float32)).item()
        h8 = torch.relu(torch.tensor(self.l1_w8 * x + self.l1_b8, dtype=torch.float32)).item()

        g1 = torch.relu(torch.tensor(
            self.l2_w11 * h1 + self.l2_w12 * h2 + self.l2_w13 * h3 + self.l2_w14 * h4 +
            self.l2_w15 * h5 + self.l2_w16 * h6 + self.l2_w17 * h7 + self.l2_w18 * h8 + self.l2_b1,
            dtype=torch.float32
        )).item()
        g2 = torch.relu(torch.tensor(
            self.l2_w21 * h1 + self.l2_w22 * h2 + self.l2_w23 * h3 + self.l2_w24 * h4 +
            self.l2_w25 * h5 + self.l2_w26 * h6 + self.l2_w27 * h7 + self.l2_w28 * h8 + self.l2_b2,
            dtype=torch.float32
        )).item()
        g3 = torch.relu(torch.tensor(
            self.l2_w31 * h1 + self.l2_w32 * h2 + self.l2_w33 * h3 + self.l2_w34 * h4 +
            self.l2_w35 * h5 + self.l2_w36 * h6 + self.l2_w37 * h7 + self.l2_w38 * h8 + self.l2_b3,
            dtype=torch.float32
        )).item()
        g4 = torch.relu(torch.tensor(
            self.l2_w41 * h1 + self.l2_w42 * h2 + self.l2_w43 * h3 + self.l2_w44 * h4 +
            self.l2_w45 * h5 + self.l2_w46 * h6 + self.l2_w47 * h7 + self.l2_w48 * h8 + self.l2_b4,
            dtype=torch.float32
        )).item()
        g5 = torch.relu(torch.tensor(
            self.l2_w51 * h1 + self.l2_w52 * h2 + self.l2_w53 * h3 + self.l2_w54 * h4 +
            self.l2_w55 * h5 + self.l2_w56 * h6 + self.l2_w57 * h7 + self.l2_w58 * h8 + self.l2_b5,
            dtype=torch.float32
        )).item()
        g6 = torch.relu(torch.tensor(
            self.l2_w61 * h1 + self.l2_w62 * h2 + self.l2_w63 * h3 + self.l2_w64 * h4 +
            self.l2_w65 * h5 + self.l2_w66 * h6 + self.l2_w67 * h7 + self.l2_w68 * h8 + self.l2_b6,
            dtype=torch.float32
        )).item()
        g7 = torch.relu(torch.tensor(
            self.l2_w71 * h1 + self.l2_w72 * h2 + self.l2_w73 * h3 + self.l2_w74 * h4 +
            self.l2_w75 * h5 + self.l2_w76 * h6 + self.l2_w77 * h7 + self.l2_w78 * h8 + self.l2_b7,
            dtype=torch.float32
        )).item()
        g8 = torch.relu(torch.tensor(
            self.l2_w81 * h1 + self.l2_w82 * h2 + self.l2_w83 * h3 + self.l2_w84 * h4 +
            self.l2_w85 * h5 + self.l2_w86 * h6 + self.l2_w87 * h7 + self.l2_w88 * h8 + self.l2_b8,
            dtype=torch.float32
        )).item()

        output = (
            self.l3_w1 * g1 + self.l3_w2 * g2 + self.l3_w3 * g3 + self.l3_w4 * g4 +
            self.l3_w5 * g5 + self.l3_w6 * g6 + self.l3_w7 * g7 + self.l3_w8 * g8 + self.l3_b
        )
        return output
    
    def loss(self, dataset: list[tuple[float, float]]):
        res = 0
        for x, expected in dataset:
            actual = self.forward(x)
            res += (actual - expected) ** 2
        return res / len(dataset)

def finite_diff(model: ModelSquare, dataset: list[tuple[float, float]], eps: float) -> ModelSquare:
    gradient = ModelSquare()

    original_loss = model.loss(dataset)

    for name in model.parameter_names():
        original_value = getattr(model, name)

        setattr(model, name, original_value + eps)
        grad_value = (model.loss(dataset) - original_loss) / eps
        setattr(gradient, name, grad_value)

        setattr(model, name, original_value)

    return gradient

def train_step(model: ModelSquare, gradient: ModelSquare, learning_rate: float) -> ModelSquare:
    new_model = copy.copy(model)

    for name in model.parameter_names():
        current_value = getattr(model, name)
        grad_value = getattr(gradient, name)
        setattr(new_model, name, current_value - learning_rate * grad_value)

    return new_model

def train(model: ModelSquare, train_data, val_data, num_epochs, learning_rate = 0.01, eps=1e-4):
    history_train = []
    history_val = []

    for epoch in range(num_epochs):
        gradient = finite_diff(model, train_data, eps)
        model = train_step(model, gradient, learning_rate)

        train_loss = model.loss(train_data)
        val_loss = model.loss(val_data)

        history_train.append(train_loss)
        history_val.append(val_loss)

        if epoch % 50 == 0:
            print(f"{epoch:05d} Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    return history_train, history_val

def predict(model: ModelSquare, xs: list[float]) -> list[float]:
    result = []
    for x in xs:
        result.append(model.forward(x))
    return result

def main():
    # keeping intial dataset generation for archival purposes
    # dataset_initial = []
    # for n in range(100):
    #     curr_set = [n, n**2]
    #     dataset_initial.append(curr_set)
    # dataset = torch.tensor(dataset_initial)

    dataset = []
    for _ in range(500):
        x = random.random() * 20.0 - 10.0
        y = x * x
        dataset.append((x, y))

    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    eps = 1e-4
    learning_rate = 1e-4
    num_epochs = 2000

    model = ModelSquare()

    history_train, history_val = train(model, train_data, val_data, num_epochs, learning_rate, eps)


    test_values = [4.0, -4.0, 11.0, 20.0, 8.0, -5.0]
    preds = predict(model, test_values)

    print("\nPredictions:")
    for x, y_pred in zip(test_values, preds):
        print(f"x={x:>5}  pred={y_pred:.6f}  true={x*x:.6f}")

    xs = list(range(len(history_train)))
    plt.plot(xs, history_train, label="Train Loss")
    plt.plot(xs, history_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Custom-weight neural net learning x^2 with finite-difference SGD")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()