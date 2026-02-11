import torch.nn as nn

def get_wine_model(input_size=13, hidden_size=16, output_size=3):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    return model
