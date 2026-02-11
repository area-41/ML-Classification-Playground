import torch
import torch.nn as nn
from src.softmax.softmax_2d.model import get_wine_model
from src.preprocessing.scaling import apply_standard_scaling

def train_model(train_loader, epochs=100):
    model = get_wine_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Salva o estado do modelo na pasta correta
    torch.save(model.state_dict(), 'models/softmax_2d_wine.pth')
    return model
