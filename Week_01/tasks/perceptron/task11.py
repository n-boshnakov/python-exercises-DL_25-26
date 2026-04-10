import matplotlib.pyplot as plt
import numpy as np
try:
    from . import task01, task06, task07, task08
except ImportError:
    import task01, task06, task07, task08

class ModelSquare():
    def __init__(self, dataset: list, input_size: int=1, hidden_size: int=1, hidden_amount: int=1):
        # use nn.linear from torch; layer 1: 1 -> 8, ...
        self.size_input = input_size
        self.size_hidden = hidden_size
        self.amount_hidden = hidden_amount
        self.weights_input = []
        self.weights_hidden = []
        self.dataset = dataset

        self.weights_input.append(self.initialize_multiple_weights(self.weights_input, self.size_input, self.size_hidden))
        print(f"Input Layer -> Hiddden Layer: {self.weights_input}")
        if (self.amount_hidden > 1):
            for n in range(self.amount_hidden - 1):
                self.weights_input.append(self.initialize_multiple_weights(self.weights_hidden, self.size_hidden, self.size_hidden))
        self.weights_input.append(self.initialize_multiple_weights(self.weights_hidden, self.size_hidden, 1))
        print(f"Hidden -> Hidden: {self.weights_hidden}")

    def initialize_multiple_weights(self, weights_list: list, input_size: int, output_size: int):
        for n in range(input_size):
            curr_tuple = []
            for num in range(output_size):
                curr_w = task01.initialize_weights(-1, 1)
                curr_tuple.append(curr_w)
            b = task01.initialize_weights(-1, 1)
            curr_tuple.append(b)
        weights_list.append(curr_tuple)
        return weights_list

# out_or = task07.sigmoid((data[0] * (self.w1_or + w1_or_eps) + data[1] * (self.w2_or + w2_or_eps)) + (self.b_or + b_or_eps))
# out_nand = task07.sigmoid((data[0] * (self.w1_nand + w1_nand_eps) + data[1] * (self.w2_nand + w2_nand_eps)) + (self.b_nand + b_nand_eps))


    # def forward_layer(self, data, input: list, output: list, input_size: int, output_size: int, weights_input: list, eps_input: list=np.zeros((1, input_size))):
    #     output_list = []
    #     for n in range(output_size):
    #         output = 0
    #         curr_num = 0
    #         for num in input:

    #             output += num * 
    #         output.append(task07.sigmoid(output))
    #     return output

    def forward(self, data, eps_input, eps_hidden, eps_output):
        pass
            
        