import torch
import os
import pandas as pd
import json
from datetime import datetime

def save_model(model, file_path):
    """Salva o modelo PyTorch no caminho especificado."""
    torch.save(model.state_dict(), file_path)
    print(f"Modelo salvo em {file_path}")

def load_model(model, file_path):
    """Carrega pesos do modelo PyTorch a partir de um arquivo."""
    model.load_state_dict(torch.load(file_path))
    print(f"Modelo carregado de {file_path}")
    return model

def save_embeddings(embeddings, file_path):
    """Salva embeddings em um arquivo CSV."""
    pd.DataFrame(embeddings.numpy()).to_csv(file_path, index=False)
    print(f"Embeddings salvos em {file_path}")

def load_embeddings(file_path):
    """Carrega embeddings de um arquivo CSV para um tensor PyTorch."""
    embeddings = torch.tensor(pd.read_csv(file_path).values)
    print(f"Embeddings carregados de {file_path}")
    return embeddings

def save_metrics(metrics, file_path):
    """Salva métricas de avaliação em um arquivo JSON."""
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas salvas em {file_path}")

def load_metrics(file_path):
    """Carrega métricas de um arquivo JSON."""
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    print(f"Métricas carregadas de {file_path}")
    return metrics

def create_timestamped_dir(base_dir="checkpoints"):
    """Cria um diretório com timestamp para armazenar arquivos."""
    dir_path = os.path.join(base_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(dir_path, exist_ok=True)
    print(f"Diretório criado: {dir_path}")
    return dir_path

def save_training_state(model, optimizer, epoch, file_path):
    """Salva o estado do treinamento (modelo, otimizador e época)."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, file_path)
    print(f"Estado de treinamento salvo em {file_path}")

def load_training_state(model, optimizer, file_path):
    """Carrega o estado do treinamento (modelo, otimizador e época)."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Estado de treinamento carregado de {file_path}")
    return checkpoint['epoch']
