import numpy as np

# https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d
class MultiLayerPerceptron():
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        # self.input_weight_hidden = np.random.randn(input_size, hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size_1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size_1))

        self.w2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2. / hidden_size_1)
        self.b2 = np.zeros((1, hidden_size_2))

        # self.output_weight = np.random.randn(hidden_size, output_size)
        self.w3 = np.random.randn(hidden_size_2, output_size) * np.sqrt(2. / hidden_size_2)
        self.b3 = np.zeros((1, output_size))
        self.x_scale = 1.0
        self.y_scale = 1.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # https://www.delftstack.com/howto/numpy/numpy-relu/
    # can be used instead of softmax
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def LeakyReLU(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def LeakyReLU_deriv(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, x, classification=False):
        self.z1 = np.dot(x, self.w1) + self.b1
        # self.hidden_output = self.sigmoid(self.hidden_input)
        # self.hidden_output = self.ReLU(self.hidden_input)
        self.a1 = self.LeakyReLU(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.LeakyReLU(self.z2)

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        if(classification):
            self.final_output = self.ReLU(self.z3)
        else:
            self.final_output = self.z3

        return self.final_output

    def backward(self, x, y, output, learning_rate):
        m = x.shape[0]  # Number of samples (for averaging)

        dz3 = (output - y)
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * self.LeakyReLU_deriv(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.LeakyReLU_deriv(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.w3 -= learning_rate * dw3
        self.b3 -= learning_rate * db3
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1


    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)
            if (epoch+1) % 100 == 0:
                loss = np.mean((y - output)**2)
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    def model(self, x):
        # Normalize input
        x_scaled = np.array(x).reshape(-1, 1).astype(float) / self.x_scale

        # Forward pass
        y_scaled = self.forward(x_scaled)

        # Denormalize output
        return y_scaled * self.y_scale

def main():
    dataset_reg = (np.arange(1, 41).astype(int)).tolist()
    dataset_squared = list(map(lambda x: int(x**2), dataset_reg))

    x = np.array(dataset_reg).reshape(-1, 1).astype(float) / 40
    y = np.array(dataset_squared).reshape(-1, 1).astype(float) / (40**2)

    multi_layer_perceptron = MultiLayerPerceptron(1, 64, 32, 1)
    multi_layer_perceptron.x_scale = 40
    multi_layer_perceptron.y_scale = 40**2

    multi_layer_perceptron.train(x, y, 100000, 0.01)

    print(f"2^2 = {multi_layer_perceptron.model(2)}")
    print(f"3^2 = {multi_layer_perceptron.model(3)}")
    print(f"5^2 = {multi_layer_perceptron.model(5)}")
    print(f"14^2 = {multi_layer_perceptron.model(14)}")
    print(f"27^2 = {multi_layer_perceptron.model(27)}")
    
if __name__ == "__main__":
    main()