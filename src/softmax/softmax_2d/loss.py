import torch

def manual_cross_entropy(predictions, targets):
    """
    Calcula a Cross Entropy Loss.
    predictions: probabilidades após o softmax (batch_size, num_classes)
    targets: labels reais em formato one-hot ou índices
    """
    samples = predictions.shape[0]
    # Pequeno valor para evitar log(0)
    eps = 1e-7
    predictions = torch.clamp(predictions, eps, 1 - eps)
    
    # Se targets forem índices, extraímos as probabilidades da classe correta
    log_likelihood = -torch.log(predictions[range(samples), targets])
    loss = torch.sum(log_likelihood) / samples
    return loss
