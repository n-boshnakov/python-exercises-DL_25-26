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

    def forward(self, data, w1_or_eps: float=0, w2_or_eps: float=0, w1_nand_eps: float=0, w2_nand_eps: float=0, w1_and_eps: float=0, w2_and_eps: float=0, b_or_eps: float=0, b_nand_eps: float=0, b_and_eps: float=0):
        out_or = (data[0] * (self.w1_or + w1_or_eps) + data[1] * (self.w2_or + w2_or_eps)) + (self.b_or + b_or_eps)
        out_nand = (data[0] * (self.w1_nand + w1_nand_eps) + data[1] * (self.w2_nand + w2_nand_eps)) + (self.b_nand + b_nand_eps)
        out_and = (data[0] * (self.w1_and + w1_and_eps) + data[1] * (self.w2_and + w2_and_eps)) + (self.b_and + b_and_eps)
        return out_and

    def calc_loss(self, w1_or_eps: float=0, w2_or_eps: float=0, w1_nand_eps: float=0, w2_nand_eps: float=0, w1_and_eps: float=0, w2_and_eps: float=0, b_or_eps: float=0, b_nand_eps: float=0, b_and_eps: float=0):
        sum = 0
        for (x, y, z) in self.dataset:
            sum += (task07.sigmoid((x * (self.w1 + w1_or_eps) + y * (self.w2 + w2_or_eps)) + (self.b_or + b_or_eps)) - z)**2
    
        return sum / len(self.dataset)
    
    def calc_derivative(self, w1_eps: float=0, w2_eps: float=0, b_eps: float=0):
        loss1 = self.calc_loss()
        loss2 = self.calc_loss(w1_eps, w2_eps, b_eps)
        self.loss_list.append((loss2 - loss1) / (w1_eps if w1_eps != 0 else w2_eps if w2_eps != 0 else b_eps))

        return (loss2 - loss1) / (w1_eps if w1_eps != 0 else w2_eps if w2_eps != 0 else b_eps)
    
    def single_step(self, learning_rate):
        print(f"Loss before: {self.calc_loss()}")
        d_w1 = self.calc_derivative(w1_eps = 0.001)
        d_w2 = self.calc_derivative(w2_eps = 0.001)
        d_b = self.calc_derivative(b_eps = 0.001)

        self.w1 -= d_w1 * learning_rate
        self.w2 -= d_w2 * learning_rate
        self.bias -= d_b * learning_rate
        print(f"Loss after: {self.calc_loss()}")
        return self.w1, self.w2

    def train(self, epochs, learning_rate=0.001):
        for n in range(epochs):
            self.single_step(learning_rate)
            pass

    def predict(self, input1, input2):
        return (task07.sigmoid((self.w1*input1 + self.w2*input2) + self.bias))
def main():
    dataset_XOR = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    modelXOR = ModelXOR(dataset_XOR)

    # bias
    print(f"XOR for 1 and 1: {modelXOR.predict(1, 1)}")
    print(f"XOR for 0 and 1: {modelXOR.predict(0, 1)}")
    print(f"XOR for 1 and 0: {modelXOR.predict(1, 0)}")
    print(f"XOR for 0 and 0: {modelXOR.predict(0, 0)}")
    
if __name__ == '__main__':
    main()