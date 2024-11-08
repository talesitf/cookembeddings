import torch
from torch import nn, optim
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_dim, latent_dim, lr=0.001):
        """
        Inicializa o modelo de autoencoder, a função de perda e o otimizador.
        
        Args:
            input_dim (int): Dimensão de entrada dos embeddings.
            latent_dim (int): Dimensão do espaço latente no encoder.
            lr (float): Taxa de aprendizado para o otimizador.
        """
        self.autoencoder = self._build_autoencoder(input_dim, latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []

    def _build_autoencoder(self, input_dim, latent_dim):
        """
        Define a arquitetura do autoencoder.
        
        Args:
            input_dim (int): Dimensão de entrada dos embeddings.
            latent_dim (int): Dimensão do espaço latente no encoder.
        
        Returns:
            nn.Module: O modelo de autoencoder.
        """
        class Autoencoder(nn.Module):
            def __init__(self):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )

            def forward(self, x):
                latent = self.encoder(x)
                reconstructed = self.decoder(latent)
                return reconstructed, latent

        return Autoencoder()

    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Treina o autoencoder e armazena as perdas para visualização.
        
        Args:
            train_loader (DataLoader): DataLoader para dados de treino.
            val_loader (DataLoader, optional): DataLoader para dados de validação.
            epochs (int): Número de épocas de treinamento.
        """
        for epoch in range(epochs):
            epoch_train_loss = 0
            self.autoencoder.train()
            
            for inputs in train_loader:
                inputs = inputs[0]  # Remove o segundo elemento do DataLoader tuple
                self.optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(inputs)
                loss = self.criterion(reconstructed, inputs)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            if val_loader:
                avg_val_loss = self.evaluate(val_loader)
                self.val_losses.append(avg_val_loss)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    def evaluate(self, dataloader):
        """
        Avalia o autoencoder em um conjunto de dados.
        
        Args:
            dataloader (DataLoader): DataLoader para dados de avaliação.
        
        Returns:
            float: Média da perda de avaliação.
        """
        self.autoencoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0]  # Remove o segundo elemento do DataLoader tuple
                reconstructed, _ = self.autoencoder(inputs)
                loss = self.criterion(reconstructed, inputs)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def plot_loss(self):
        """Visualiza as perdas de treino e validação ao longo das épocas."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    def get_compressed_embeddings(self, data_loader):
        """
        Extrai os embeddings comprimidos do encoder.
        
        Args:
            data_loader (DataLoader): DataLoader para os dados de entrada.
        
        Returns:
            torch.Tensor: Embeddings comprimidos.
        """
        self.autoencoder.eval()
        compressed_embeddings = []
        
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0]  # Remove o segundo elemento do DataLoader tuple
                _, latent = self.autoencoder(inputs)
                compressed_embeddings.append(latent)
        
        return torch.cat(compressed_embeddings, dim=0)
