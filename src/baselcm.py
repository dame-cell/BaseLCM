import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm.auto import tqdm
import torch

class SonarEncoder:
    """
    SONAR Encoder: Encodes sentences into embeddings using the SONAR model, with support for batching.
    """
    def __init__(self, model_name='cointegrated/SONAR_200_text_encoder', device="cpu"):
        self.encoder = M2M100Encoder.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def encode(self, texts, lang, batch_size=32, norm=False):
        """
        Encode texts into embeddings with batching and optional normalization.

        Args:
            texts (List[str]): List of input texts to encode.
            lang (str): Language code for the tokenizer.
            batch_size (int): Number of texts to process in a single batch.
            norm (bool): Whether to normalize the embeddings.

        Returns:
            torch.Tensor: Encoded embeddings.
        """
        if self.tokenizer is None or self.encoder is None:
            raise ValueError("Tokenizer or encoder is not initialized.")

        self.tokenizer.src_lang = lang
        texts = texts if isinstance(texts, list) else [texts]

        embeddings = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches", unit="batch"):
                batch_texts = texts[i:i + batch_size]
                batch = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                seq_embs = self.encoder(**batch).last_hidden_state
                mask = batch.attention_mask

                # Compute mean embedding for each sequence
                mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
                if norm:
                    mean_emb = torch.nn.functional.normalize(mean_emb, dim=1)

                embeddings.append(mean_emb)

        return torch.cat(embeddings, dim=0)

  
class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None
    
    def fit(self, embeddings):
        # Calculate median and IQR along each dimension
        self.median = torch.median(embeddings, dim=0)[0]
        q75, q25 = torch.quantile(embeddings, torch.tensor([0.75, 0.25]), dim=0)
        self.iqr = q75 - q25
        # Avoid division by zero
        self.iqr = torch.where(self.iqr == 0, torch.ones_like(self.iqr), self.iqr)
    
    def normalize(self, x):
        return (x - self.median) / self.iqr
    
    def denormalize(self, x):
        return self.median + (x * self.iqr)

class PreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.scaler = RobustScaler()
    
    def fit_scaler(self, sample_embeddings):
        self.scaler.fit(sample_embeddings)
    
    def forward(self, x):
        x = self.scaler.normalize(x)
        x = self.linear(x)
        return x

class PostNet(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.scaler = None  # Will be shared with PreNet
    
    def forward(self, x):
        x = self.linear(x)
        x = self.scaler.denormalize(x)
        return x
      
class TransformerDecoder(nn.Module):
    def __init__(self,hidden_dim,num_heads,num_layers,ff_dim,dropout=0.1,max_seq_len=512):
        super(TransformerDecoder,self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
      
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

    def forward(self,x):
        seq_len = x.size(1)
        # Ensure we don't exceed the input sequence length
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        for layer in self.layers:
            x = layer(x,x)
        return x
    
class BaseLCM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim):
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer_decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
        self.postnet = PostNet(hidden_dim, output_dim)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)
        return x.squeeze(1)  # Remove sequence dimension if single step