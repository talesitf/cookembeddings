import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_processing import DataProcessor
from models.autoencoder import Autoencoder
from src.utils import save_model, load_model, create_timestamped_dir, save_training_state, load_training_state
from src.config import *

# Inicializa DataProcessor e carrega os dados
data_processor = DataProcessor(model_name=MODEL_NAME, file_path='./data/13k-recipes.csv')
df = data_processor.df
texts = data_processor.preprocess_data(df, 'Instructions')
# Gerar embeddings com barra de progresso
embeddings = data_processor.generate_embeddings(texts, batch_size=32)
train_embeddings, test_embeddings = data_processor.split_data(embeddings)

# Preparar DataLoader
train_loader = DataLoader(TensorDataset(train_embeddings), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(test_embeddings), batch_size=BATCH_SIZE)

# Inicializar o Autoencoder e Treinamento
trainer = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, lr=LEARNING_RATE)

# Criar diretório para checkpoints
checkpoint_dir = create_timestamped_dir()

# Salvar estado de treinamento
trainer.train(train_loader, val_loader=test_loader, epochs=EPOCHS)  # Treinamento em uma época por vez
save_training_state(trainer.autoencoder, trainer.optimizer, epoch + 1, f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt")

# Salvar o modelo final
save_model(trainer.autoencoder, f"{checkpoint_dir}/final_model.pt")