import torch
import joblib
import numpy as np
import os
from src.softmax.softmax_2d.model import get_wine_model

class WineInferenceEngine:
    def __init__(self, model_path='model/classifier.pth', scaler_path='model/scaler.pkl'):
        """
        Inicializa o motor de inferência carregando o modelo e o normalizador.
        """
        # 1. Definir o dispositivo (CPU ou GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Carregar o Scaler (essencial para manter a acurácia)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # 3. Carregar a Arquitetura e Pesos
        self.model = get_wine_model()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pesos do modelo não encontrados em: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Modo de avaliação desativa dropout/batchnorm

    def predict(self, raw_features):
        """
        Realiza a predição para uma lista de características.
        raw_features: lista ou array com 13 valores (teor alcoólico, etc)
        """
        # Converter para array 2D e normalizar
        features_arr = np.array(raw_features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_arr)
        
        # Converter para tensor
        input_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Extrair resultados
            conf, pred = torch.max(probabilities, dim=1)
            
        return {
            "class_id": int(pred.item()),
            "confidence": float(conf.item()),
            "all_probabilities": probabilities.cpu().numpy().tolist()[0]
        }

# --- Exemplo de uso para o seu Notebook ou GitHub ---
if __name__ == "__main__":
    # Exemplo de dados brutos (Vinho da Classe 1)
    sample_wine = [12.3, 1.7, 2.1, 19.0, 80.0, 1.8, 2.0, 0.3, 1.3, 3.0, 1.0, 2.8, 500.0]
    
    try:
        engine = WineInferenceEngine()
        result = engine.predict(sample_wine)
        
        print(f"--- Resultado da Inferência ---")
        print(f"Classe Predita: {result['class_id']}")
        print(f"Confiança: {result['confidence']:.4f}")
        print(f"Probabilidades: {result['all_probabilities']}")
        
    except Exception as e:
        print(f"Erro na inferência: {e}")
