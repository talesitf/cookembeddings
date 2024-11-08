import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2', file_path: str = None):
        """
        Inicializa o processador de dados com o modelo de embeddings sBERT.
        
        Args:
            model_name (str): Nome do modelo sBERT a ser usado para gerar embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.df = self._load_data(file_path) if file_path else None

    def _load_data(self, file_path):
        """
        Carrega o DataFrame a partir de um arquivo CSV.
        
        Args:
            file_path (str): Caminho para o arquivo de dados.
        
        Returns:
            pd.DataFrame: DataFrame contendo os dados carregados.
        """
        df = pd.read_csv(file_path)
        return df

    def preprocess_data(self, df, text_column):
        """
        Extrai a coluna de texto e gera uma lista de textos para o modelo.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados.
            text_column (str): Nome da coluna contendo o texto para gerar embeddings.
        
        Returns:
            list of str: Lista de textos para geração de embeddings.
        """
        texts = df[text_column].tolist()

        return texts

    def generate_embeddings(self, texts, batch_size=32):
        """
        Gera embeddings para uma lista de textos, com exibição de progresso.
        
        Args:
            texts (list of str): Lista de textos para os quais queremos gerar embeddings.
            batch_size (int): Tamanho do lote de textos a serem processados simultaneamente.
        
        Returns:
            torch.Tensor: Tensor com os embeddings gerados.
        """
        embeddings = []
        
        # Divide os textos em lotes e mostra a barra de progresso
        for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return torch.tensor(embeddings)

    def split_data(self, embeddings, test_size=0.2, random_state=42):
        """
        Divide os dados de embeddings em conjuntos de treino e teste.
        
        Args:
            embeddings (torch.Tensor): Embeddings dos textos.
            test_size (float): Proporção de dados para o conjunto de teste.
            random_state (int): Semente para garantir reprodutibilidade.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Embeddings de treino e teste.
        """
        train_embeddings, test_embeddings = train_test_split(
            embeddings, test_size=test_size, random_state=random_state
        )
        return torch.tensor(train_embeddings), torch.tensor(test_embeddings)
