import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(train_losses, val_accuracies):
    """Gera gráficos de perda e acurácia lado a lado."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(train_losses, label='Training Loss', color='royalblue')
    ax[0].set_title('Evolução da Perda')
    ax[0].set_xlabel('Época')
    ax[0].legend()
    
    ax[1].plot(val_accuracies, label='Validation Accuracy', color='forestgreen')
    ax[1].set_title('Evolução da Acurácia')
    ax[1].set_xlabel('Época')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()
