"""
# Trecho de teste conceitual
output = model(input_data)
output.backward() # Gradiente do PyTorch

# Comparação
my_grad_w, my_grad_b = compute_gradients(input_data, torch.softmax(output, dim=1), labels)

print(f"Diferença nos pesos: {torch.abs(model[0].weight.grad.t() - my_grad_w).mean()}")
"""

import torch
import numpy as np
import pytest
from src.softmax.softmax_2d.gradients import compute_gradients
from src.softmax.softmax_2d.model import get_wine_model

def test_gradient_consistency():
    """
    Testa se o gradiente manual é idêntico ao gradiente do PyTorch.
    """
    # 1. Setup: Criar dados sintéticos e modelo
    torch.manual_seed(42)
    inputs = torch.randn(5, 13) # Batch de 5, 13 features
    labels = torch.tensor([0, 1, 2, 0, 1])
    
    # Criamos uma camada linear simples para testar o gradiente
    # (Em testes de gradiente, isolamos a camada para evitar ruído de ReLUs)
    linear_layer = torch.nn.Linear(13, 3)
    
    # 2. PyTorch Autograd
    outputs = linear_layer(inputs)
    probs = torch.softmax(outputs, dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    loss.backward()
    
    torch_grad_w = linear_layer.weight.grad
    torch_grad_b = linear_layer.bias.grad

    # 3. Nosso cálculo manual (gradients.py)
    # Note: O PyTorch guarda os pesos como (out_features, in_features), 
    # por isso o transpose no final do nosso cálculo manual se necessário.
    manual_grad_w, manual_grad_b = compute_gradients(inputs, probs, labels)

    # 4. Asserção (Comparação)
    # Verificamos se a diferença média é quase zero (tolerância de float)
    assert torch.allclose(torch_grad_w, manual_grad_w.t(), atol=1e-6), "Gradiente de pesos diverge!"
    assert torch.allclose(torch_grad_b, manual_grad_b, atol=1e-6), "Gradiente de bias diverge!"

    print("\n Teste de consistência de gradientes passou!")

if __name__ == "__main__":
    test_gradient_consistency()
