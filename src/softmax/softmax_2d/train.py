"""
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
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os

# Importações internas
from src.softmax.softmax_2d.model import get_wine_model
from src.preprocessing.scaling import apply_standard_scaling

def run_training(epochs=100, batch_size=16, lr=0.01):
    print("--- Iniciando Laboratório de Treinamento: Softmax 2D ---")
    
    # 1. Carregamento de Dados
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Pré-processamento e Salvamento do Scaler
    os.makedirs('model', exist_ok=True)
    X_train_scaled, _ = apply_standard_scaling(X_train, save_path='model/scaler.pkl')
    X_test_scaled, _ = apply_standard_scaling(X_test) # Apenas para validação rápida
    
    # 3. Preparação para PyTorch
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # 4. Inicialização do Modelo, Perda e Otimizador
    model = get_wine_model(input_size=13, hidden_size=16, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Loop de Treinamento
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Época [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(train_loader):.4f}")
    
    # 6. Salvamento dos Pesos do Modelo
    torch.save(model.state_dict(), 'model/classifier.pth')
    print("✅ Treinamento finalizado. Artefatos salvos em /model")

if __name__ == "__main__":
    run_training()
