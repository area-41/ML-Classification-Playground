import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from src.softmax.softmax_2d.model import get_wine_model

def run_kfold_validation(X, y, k=5, epochs=50):
    """
    Executa a validação cruzada K-Fold para garantir robustez.
    """
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Criar loaders para este fold específico
        train_sub = Subset(dataset, train_ids)
        test_sub = Subset(dataset, test_ids)
        
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_sub, batch_size=16, shuffle=False)
        
        # Reinicializar o modelo para cada fold
        model = get_wine_model()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Loop de treino simplificado
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
        
        # Avaliação do fold
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        acc = correct / len(test_ids)
        results.append(acc)
        print(f"Fold {fold+1}: Acurácia = {acc*100:.2f}%")
        
    print(f"\nAcurácia Média Final: {np.mean(results)*100:.2f}% (+/- {np.std(results)*100:.2f}%)")
    return results
