import torch

def compute_gradients(inputs, probabilities, targets):
    """
    Cálculo manual do gradiente para uma camada linear + softmax.
    dL/dW = (P - Y) * X
    """
    batch_size = inputs.shape[0]
    
    # Converter targets para one-hot encoding
    num_classes = probabilities.shape[1]
    targets_one_hot = torch.zeros_like(probabilities)
    targets_one_hot.scatter_(1, targets.view(-1, 1), 1)
    
    # A diferença (Erro)
    error = probabilities - targets_one_hot # (P - Y)
    
    # Gradiente para os Pesos (Weights)
    # Matriz Transposta de Input x Erro
    grad_weights = torch.matmul(inputs.t(), error) / batch_size
    
    # Gradiente para o Viés (Bias)
    grad_bias = torch.sum(error, dim=0) / batch_size
    
    return grad_weights, grad_bias
