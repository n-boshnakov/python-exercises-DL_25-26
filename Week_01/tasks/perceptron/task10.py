import matplotlib.pyplot as plt
try:
    from . import task01, task06, task07, task08, task09
except ImportError:
    import task01, task06, task07, task08, task09


class ModelXOR():
    def __init__(self, dataset, w1_or=task01.initialize_weights(-1, 1), w2_or=task01.initialize_weights(-1, 1), w1_nand=task01.initialize_weights(-1, 1), w2_nand=task01.initialize_weights(-1, 1), w1_and=task01.initialize_weights(-1, 1), w2_and=task01.initialize_weights(-1, 1)):
        self.w1_or = w1_or
        self.w2_or = w2_or
        self.w1_nand = w1_nand
        self.w2_nand = w2_nand
        self.w1_and = w1_and
        self.w2_and = w2_and
        self.b_or = task01.initialize_weights(-1, 1)
        self.b_nand = task01.initialize_weights(-1, 1)
        self.b_and = task01.initialize_weights(-1, 1)
        self.dataset = dataset

    def forward(self, data: list, w1_or_eps: float=0, w2_or_eps: float=0, w1_nand_eps: float=0, w2_nand_eps: float=0, w1_and_eps: float=0, w2_and_eps: float=0, b_or_eps: float=0, b_nand_eps: float=0, b_and_eps: float=0):
        out_or = task07.sigmoid((data[0] * (self.w1_or + w1_or_eps) + data[1] * (self.w2_or + w2_or_eps)) + (self.b_or + b_or_eps))
        out_nand = task07.sigmoid((data[0] * (self.w1_nand + w1_nand_eps) + data[1] * (self.w2_nand + w2_nand_eps)) + (self.b_nand + b_nand_eps))
        out_and = task07.sigmoid((out_or * (self.w1_and + w1_and_eps) + out_nand * (self.w2_and + w2_and_eps)) + (self.b_and + b_and_eps))
        return out_and

    def calc_loss(self, w1_or_eps: float=0, w2_or_eps: float=0, w1_nand_eps: float=0, w2_nand_eps: float=0, w1_and_eps: float=0, w2_and_eps: float=0, b_or_eps: float=0, b_nand_eps: float=0, b_and_eps: float=0):
        sum = 0
        for (x, y, z) in self.dataset:
            pred_y = self.forward([x, y], w1_or_eps, w2_or_eps, w1_nand_eps, w2_nand_eps, w1_and_eps, w2_and_eps, b_or_eps, b_nand_eps, b_and_eps)
            sum += (pred_y - z)**2
        return sum / len(self.dataset)

    def train(self, epochs, learning_rate=0.1):
        eps = 0.1
        for n in range(epochs):
            current_loss = self.calc_loss()
            print(f'{n:05d} Loss: {current_loss}')
            d_or_w1 = (self.calc_loss(w1_or_eps=eps) - current_loss) / eps
            self.w1_or -= learning_rate * d_or_w1
            d_or_w2 = (self.calc_loss(w2_or_eps=eps) - current_loss) / eps
            self.w2_or -= learning_rate * d_or_w2
            d_or_b = (self.calc_loss(b_or_eps=eps) - current_loss) / eps
            self.b_or -= learning_rate * d_or_b
            d_nand_w1 = (self.calc_loss(w1_nand_eps=eps) - current_loss) / eps
            self.w1_nand -= learning_rate * d_nand_w1
            d_nand_w2 = (self.calc_loss(w2_nand_eps=eps) - current_loss) / eps
            self.w2_nand -= learning_rate * d_nand_w2
            d_nand_b = (self.calc_loss(b_nand_eps=eps) - current_loss) / eps
            self.b_nand -= learning_rate * d_nand_b
            d_and_w1 = (self.calc_loss(w1_and_eps=eps) - current_loss) / eps
            self.w1_and -= learning_rate * d_and_w1
            d_and_w2 = (self.calc_loss(w2_and_eps=eps) - current_loss) / eps
            self.w1_and -= learning_rate * d_and_w2
            d_and_b = (self.calc_loss(b_and_eps=eps) - current_loss) / eps
            self.b_and -= learning_rate * d_and_b


    def predict(self, input1, input2):
        return self.forward([input1, input2])
    
def main():
    dataset_XOR = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    modelXOR = ModelXOR(dataset_XOR)
    modelXOR.train(100_000)

    # bias
    print(f"XOR for 1 and 1: {modelXOR.predict(1, 1)}")
    print(f"XOR for 0 and 1: {modelXOR.predict(0, 1)}")
    print(f"XOR for 1 and 0: {modelXOR.predict(1, 0)}")
    print(f"XOR for 0 and 0: {modelXOR.predict(0, 0)}")
    
if __name__ == '__main__':
    main()