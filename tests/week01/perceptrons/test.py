import torch
import torch.nn as nn

linear_layer = nn.Linear(in_features=1, out_features=2)

user_data_tensor = torch.tensor([[0.3471]])
output = linear_layer(user_data_tensor)
print(output)

